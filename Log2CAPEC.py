import re
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional, Union
import os
import shutil
import torch
import numpy as np
import pandas as pd
import binascii
import traceback
import ast
from tqdm import tqdm
from collections import defaultdict

# Librerie AI & NLP
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.config import Settings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
from rank_bm25 import BM25Okapi

# ==============================================================================
# 1. CONFIGURAZIONE INIZIALE E COSTANTI
# ==============================================================================

# --- MODELLI SOTA (STATE OF THE ART) 2025 ---
EMBEDDING_MODEL = 'BAAI/bge-m3'
CROSS_ENCODER_MODEL = 'BAAI/bge-reranker-v2-m3'
LLM_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# PATHS
CHROMA_DB_PATH = "./chroma_db_bge_m3_final_v7_complete" 
CAPEC_XML_FILE = "CAPEC.xml"
HONEYPOT_CSV_FILE = "tpot_less.csv"

# NLP SETUP
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("\033[1;33m[CONFIG] Download modello Spacy...\033[0m")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("\033[1;33m[CONFIG] Download dati NLTK...\033[0m")
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
NS = {'capec': 'http://capec.mitre.org/capec-3', 'xhtml': 'http://www.w3.org/1999/xhtml'}

ACRONYM_MAP = {"XSS": "cross site scripting", "SQLi": "sql injection", "CSRF": "cross site request forgery",
               "RCE": "remote code execution", "LFI": "local file inclusion", "XXE": "xml external entity injection",
               "SSRF": "server-side request forgery", "CWE": "common weakness enumeration"}
TECHNICAL_SYNONYMS = {'compromise': ['breach', 'infiltrate'], 'exploit': ['vulnerability', 'weakness'],
                      'payload': ['malicious code', 'attack vector'],
                      'injection': ['code insertion', 'command insertion'], 'privilege': ['permission', 'access right']}

# ==============================================================================
# 2. FUNZIONI DI PRE-ELABORAZIONE DEL TESTO
# ==============================================================================

def expand_technical_terms(text: str) -> str:
    if not isinstance(text, str): text = str(text)
    for term, synonyms in TECHNICAL_SYNONYMS.items():
        for syn in synonyms:
            text = re.sub(rf'\b{syn}\b', term, text, flags=re.IGNORECASE)
    return text

def get_enhanced_tokens(text: str) -> List[str]:
    """Tokenizzazione ottimizzata per BM25: mantiene comandi shell e path."""
    if not isinstance(text, str): return []
    doc = nlp(text)
    # Manteniamo token anche se corti se sono simboli tecnici (es. ; | / )
    tokens = [token.text.lower() for token in doc if not token.is_stop] 
    return tokens

def preprocess_text(text: str) -> str:
    if not isinstance(text, str): return str(text)
    text = text.lower()
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', ' ip_addr ', text)
    # Preservation of shell operators and path separators
    text = re.sub(r'[^\w\s\-\./\\:;|&=<>]', ' ', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_unwanted_terms(text: str) -> str:
    if not isinstance(text, str): return str(text)
    text = re.sub(r'\b(capec|cwe)\s*\d*\b', '', text, flags=re.IGNORECASE)
    # Lista ridotta per non cancellare verbi tecnici importanti
    generic_terms = ['description', 'summary', 'details', 'content', 'reference', 'related', 'introduction', 'background']
    for term in generic_terms:
        text = re.sub(rf'\b{term}\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def expand_acronyms(text: str) -> str:
    if not isinstance(text, str): return str(text)
    for acronym, full in ACRONYM_MAP.items():
        text = re.sub(rf'\b{acronym}\b', full, text, flags=re.IGNORECASE)
    return text

def full_preprocess(text: str) -> str:
    if not isinstance(text, str): return ""
    processed = expand_technical_terms(expand_acronyms(text))
    processed = preprocess_text(processed)
    processed = remove_unwanted_terms(processed)
    processed = re.sub(r'\s+', ' ', processed).strip()
    return processed

# ==============================================================================
# 3. FUNZIONI DI ELABORAZIONE DEI DATI DEGLI HONEYPOT
# ==============================================================================

def decode_payload(hex_payload: Optional[str]) -> Tuple[str, str]:
    empty_md5 = 'd41d8cd98f00b204e9800998ecf8427e'
    empty_sha512_start = 'cf83e1357eefb8bdf1542850d66d8007'
    if pd.isna(hex_payload): return "", "empty"
    hex_payload_str = str(hex_payload).strip()
    if not hex_payload_str or hex_payload_str == empty_md5 or hex_payload_str.startswith(empty_sha512_start): return "", "empty"
    try:
        hex_payload_clean = re.sub(r'[^0-9a-fA-F]', '', hex_payload_str)
        if not hex_payload_clean or len(hex_payload_clean) % 2 != 0: return f"[Invalid Hex: {hex_payload_str[:50]}...]", "error"
        decoded_bytes = binascii.unhexlify(hex_payload_clean)
        payload_len = len(decoded_bytes)
        if payload_len == 0: return "", "empty"
        try:
            text = decoded_bytes.decode('utf-8', errors='strict')
            content_type = "text"
            non_printable = sum(1 for byte in decoded_bytes if not (32 <= byte <= 126 or byte in [9, 10, 13]))
            if non_printable / payload_len > 0.3 and payload_len > 8: content_type = "text_mixed_or_binary"
        except UnicodeDecodeError:
            try:
                text = decoded_bytes.decode('latin-1')
                content_type = "text_mixed"
            except UnicodeDecodeError:
                return f"[Binary Decode Error: {payload_len} bytes, starts {hex_payload_clean[:16]}...]", "error"
        clean_text = ''.join(filter(lambda x: 32 <= ord(x) <= 126 or x in '\n\r\t', text))
        clean_text = re.sub(r'(.+)(\r?\n\1)+', r'\1\n(repeated line)', clean_text)
        clean_text = re.sub(r'(.)\1{20,}', r'\1' * 10 + '...', clean_text)
        return clean_text[:250], content_type
    except (binascii.Error, ValueError, TypeError) as e:
        return f"[Hex Decode Error: {e} - Input: {hex_payload_str[:50]}...]", "error"
    except Exception as e_gen:
        return f"[Unknown Decode Error: {e_gen} - Input: {hex_payload_str[:50]}...]", "error"

def safe_parse_list_str(list_str: Optional[str]) -> Optional[List[str]]:
    if not list_str or pd.isna(list_str) or not isinstance(list_str, str): return None
    try:
        parsed_list = ast.literal_eval(list_str)
        if isinstance(parsed_list, list): return [str(item) for item in parsed_list]
        return None
    except (ValueError, SyntaxError, TypeError):
        return None

def generate_llm_input(log_entry_or_session: Union[Dict, List[Dict]], honeypot_type: str) -> str:
    desc_parts = []
    detailed_activity = ""

    def get_port_str(port_val):
        try:
            port_num = int(port_val)
            if port_num <= 0: return "an Unknown Port"
            return f"destination port {port_num}"
        except (ValueError, TypeError):
            return "an Unknown Port"

    if honeypot_type == 'Honeytrap':
        session_entries = log_entry_or_session if isinstance(log_entry_or_session, list) else [log_entry_or_session]
        total_events = len(session_entries)
        targeted_ports = set()
        protocols_used = set()
        payload_events_count = 0
        payload_hints = set()
        payload_descriptions = []

        for entry in session_entries:
            targeted_ports.add(get_port_str(entry.get('dest_port')))
            protocols_used.add(str(entry.get('attack_connection.protocol', 'tcp')).upper())
            payload_hex = entry.get('attack_connection.payload.data_hex', '')
            payload_len = int(entry.get('attack_connection.payload.length', 0)) if pd.notna(entry.get('attack_connection.payload.length')) else 0

            if payload_len > 0 and payload_hex:
                payload_events_count += 1
                decoded_text, content_type = decode_payload(payload_hex)
                if content_type == "error":
                    payload_descriptions.append(f"Unintelligible binary data ({payload_len} bytes)")
                elif decoded_text:
                    payload_descriptions.append(decoded_text)
                try:
                    hex_payload_clean = re.sub(r'[^0-9a-fA-F]', '', str(payload_hex))
                    if len(hex_payload_clean) > 1 and len(hex_payload_clean) % 2 == 0:
                        decoded_bytes = binascii.unhexlify(hex_payload_clean)
                        if decoded_bytes.startswith(b'\x16\x03'):
                            payload_hints.add("TLS Handshake")
                        elif any(decoded_bytes.startswith(m) for m in [b'GET ', b'POST ', b'HEAD ']):
                            payload_hints.add("HTTP Request")
                        elif decoded_bytes.startswith(b'\x03\x00'):
                            payload_hints.add("TPKT/RDP")
                except Exception: pass

        protocol_str = '/'.join(sorted(list(protocols_used)))
        ports_str = ", ".join(sorted(list(targeted_ports)))
        desc_parts.append(f"Observed {total_events} {protocol_str} connection attempts from a single source IP, targeting {len(targeted_ports)} distinct port(s): {ports_str}.")
        if payload_events_count > 0:
            desc_parts.append(f"Out of these, {payload_events_count} connection(s) contained data payloads.")
            if payload_hints:
                detailed_activity += f" Inferred Payload Types: {json.dumps(sorted(list(payload_hints)))}."
            if payload_descriptions:
                unique_descriptions = list(dict.fromkeys(payload_descriptions))
                detailed_activity += f" Example Payloads: {json.dumps(unique_descriptions[:2])}{'...' if len(unique_descriptions) > 2 else ''}."
            if detailed_activity:
                desc_parts.append(f"Specific observations:{detailed_activity}")
        else:
            desc_parts.append("This appears to be a port scanning activity with no data exchanged.")

    elif honeypot_type == 'Cowrie':
        session_entries = log_entry_or_session
        if not session_entries: return "Empty Cowrie session."
        connection_ports, connection_protocols, architectures_seen = set(), set(), set()
        tunnel_requests, downloads_info, uploads_info = [], [], []

        for event in session_entries:
            if event.get('eventid') == 'cowrie.session.connect':
                port = event.get('dest_port')
                if port and int(port) > 0: connection_ports.add(int(port))
                proto = event.get('protocol')
                if proto and isinstance(proto, str): connection_protocols.add(proto.lower())
            if event.get('eventid') == 'cowrie.session.params' and pd.notna(event.get('arch')) and event.get('arch'):
                arch_str = str(event.get('arch')).strip()
                if arch_str: architectures_seen.add(arch_str)
            if event.get('eventid') == 'cowrie.direct-tcpip.request':
                msg = event.get('message', '')
                match = re.search(r"request to (?P<host>[\w\.-]+):(?P<port>\d+)", msg)
                if match: tunnel_requests.append(f"{match.group('host')}:{match.group('port')}")
            if event.get('eventid') == 'cowrie.session.file_download':
                url = event.get('url', '[unknown URL]')
                dest_path = event.get('destfile', '[unknown destination]')
                downloads_info.append(f"Downloaded from '{url}' to '{dest_path}'")
            if event.get('eventid') == 'cowrie.session.file_upload':
                filename = event.get('filename', '[unknown file]')
                dest_path = event.get('destfile', '[unknown destination]')
                uploads_info.append(f"Uploaded file '{filename}' to '{dest_path}'")

        protocol_str = '/'.join(sorted(list(connection_protocols or {'ssh'}))).upper()
        dest_ports_str = f"destination port(s) {', '.join(map(str, sorted(list(connection_ports))))}" if connection_ports else f"default port {'22' if 'ssh' in protocol_str.lower() else '23'}"
        description_line = f"Interactive {protocol_str} session detected on {dest_ports_str}"
        if architectures_seen: description_line += f", emulating architecture(s): '{', '.join(sorted(list(architectures_seen)))}'."
        else: description_line += "."
        desc_parts.append(description_line)
        successful_login = next((e for e in session_entries if e.get('eventid') == 'cowrie.login.success'), None)
        failed_logins = [e for e in session_entries if e.get('eventid') == 'cowrie.login.failed']

        if successful_login:
            user = successful_login.get('username', 'N/A')
            if failed_logins: desc_parts.append(f"After {len(failed_logins)} failed attempt(s), login succeeded as user '{user}'.")
            else: desc_parts.append(f"Login successful as user '{user}'.")
        elif failed_logins:
            desc_parts.append(f"{len(failed_logins)} failed login attempt(s) were detected.")
            detailed_activity += f" Example username attempted: '{failed_logins[0].get('username', 'N/A')}'."
        else:
            desc_parts.append("Connection established but closed before any successful or failed login attempt was recorded.")

        commands_raw = [e.get('input', '') for e in session_entries if e.get('eventid') in ['cowrie.command.input', 'cowrie.command.failed'] and pd.notna(e.get('input'))]
        commands = [cmd.strip() for cmd in commands_raw if cmd.strip()]
        version_info = next((e.get('version', '') for e in session_entries if e.get('eventid') == 'cowrie.client.version' and e.get('version')), None)
        activity_found = False

        if commands:
            desc_parts.append(f"{len(commands)} command(s) were attempted/executed in total.")
            MAX_CMDS_SUMMARY = 15
            if len(commands) <= MAX_CMDS_SUMMARY:
                summary_commands = {"Commands sequence": commands}
            else:
                initial_cmds, final_cmds, middle_cmds = commands[:5], commands[-5:], commands[5:-5]
                counts = defaultdict(int)
                for cmd in middle_cmds: counts[cmd] += 1
                unique_cmds = [cmd for cmd in middle_cmds if counts[cmd] < 3]
                complex_pattern = re.compile(r"[|;&<>]|`|\$\(")
                complex_cmds = [cmd for cmd in middle_cmds if complex_pattern.search(cmd)]
                noteworthy_cmds = list(dict.fromkeys(unique_cmds + complex_cmds))
                summary_commands = {"Initial Commands": initial_cmds, "Noteworthy Commands from Middle": noteworthy_cmds[:5], "Final Commands": final_cmds}
            detailed_activity += f" Command Summary: {json.dumps(summary_commands)}."
            activity_found = True

        if tunnel_requests:
            desc_parts.append("The session was also used to attempt proxy/tunneling.")
            unique_tunnels = sorted(list(set(tunnel_requests)))
            detailed_activity += f" Tunnel targets observed: {json.dumps(unique_tunnels[:3])}{'...' if len(unique_tunnels) > 3 else ''}."
            tunnel_data_event = next((e for e in session_entries if e.get('eventid') == 'cowrie.direct-tcpip.data' and e.get('data')), None)
            if tunnel_data_event:
                try:
                    data_bytes = ast.literal_eval(tunnel_data_event['data'])
                    if isinstance(data_bytes, bytes) and data_bytes.startswith(b'\x16\x03'): detailed_activity += " A TLS handshake was initiated through the tunnel."
                except Exception: pass
            activity_found = True

        if uploads_info:
            desc_parts.append("File upload activity was recorded.")
            detailed_activity += f" Upload Details: {json.dumps(uploads_info[:3])}{'...' if len(uploads_info) > 3 else ''}."
            activity_found = True
        if downloads_info:
            desc_parts.append("File download activity was recorded.")
            detailed_activity += f" Download Details: {json.dumps(downloads_info[:3])}{'...' if len(downloads_info) > 3 else ''}."
            activity_found = True
        if version_info:
            detailed_activity += f" Client version string was explicitly logged: '{version_info}'."
            activity_found = True
        if not activity_found:
             desc_parts.append("No significant post-login activity was recorded.")
        if detailed_activity.strip():
            desc_parts.append(f"Specific observations:{detailed_activity.strip()}")

    elif honeypot_type == 'Dionaea':
        session_entries = log_entry_or_session if isinstance(log_entry_or_session, list) else [log_entry_or_session]
        total_events = len(session_entries)
        targeted_services, credential_captures, ftp_sessions_summary = set(), [], []

        for entry in session_entries:
            dest_port_str = get_port_str(entry.get('dest_port'))
            protocol_name = entry.get('connection.protocol', 'unknown_service')
            targeted_services.add(f"'{protocol_name}' on {dest_port_str}")
            username, password = entry.get('username'), entry.get('password')
            if pd.notna(username) or pd.notna(password):
                user_str = str(username) if pd.notna(username) else "N/A"
                credential_captures.append(f"service: {protocol_name}, user: '{user_str}'")
            if protocol_name == 'ftpd':
                ftp_cmds = safe_parse_list_str(entry.get('ftp.commands.command'))
                if ftp_cmds:
                    ftp_args = safe_parse_list_str(entry.get('ftp.commands.arguments'))
                    ftp_log_summary = [f"{cmd} {arg}".strip() for i, cmd in enumerate(ftp_cmds) for arg in ([ftp_args[i]] if ftp_args and i < len(ftp_args) else [''])]
                    ftp_sessions_summary.append(json.dumps(ftp_log_summary))

        unique_services_str = ", ".join(sorted(list(targeted_services)))
        desc_parts.append(f"Observed {total_events} connection attempts from a single source IP, targeting {len(targeted_services)} distinct service(s): {unique_services_str}.")
        activity_found = False
        if credential_captures:
            unique_creds = sorted(list(set(credential_captures)))
            detailed_activity = f" Credentials were captured during interactions. Summary: {json.dumps(unique_creds[:5])}{'...' if len(unique_creds) > 5 else ''}."
            desc_parts.append(f"Specific observations:{detailed_activity}")
            activity_found = True
        if ftp_sessions_summary:
            detailed_activity = f" Significant FTP interaction was recorded. Example command sequence: {ftp_sessions_summary[0]}{'...' if len(ftp_sessions_summary) > 1 else ''}."
            if not activity_found:
                desc_parts.append(f"Specific observations:{detailed_activity}")
            activity_found = True
        if not activity_found:
            desc_parts.append("The activity appears to be a broad service discovery scan with no further significant interaction recorded.")

    elif honeypot_type == 'Sentrypeer':
        session_entries = log_entry_or_session if isinstance(log_entry_or_session, list) else [log_entry_or_session]
        total_events = len(session_entries)
        sip_methods, user_agents, called_numbers = set(), set(), []

        for entry in session_entries:
            sip_methods.add(str(entry.get('sip_method', 'UNKNOWN')).upper())
            user_agents.add(str(entry.get('sip_user_agent', 'NOT_FOUND')))
            if pd.notna(entry.get('called_number')): called_numbers.append(str(entry.get('called_number')))

        unique_numbers = list(dict.fromkeys(called_numbers))
        methods_str = ", ".join(sorted(list(sip_methods)))
        desc_parts.append(f"Observed {total_events} SIP interaction(s) from a single source IP, using method(s): {methods_str}.")
        unique_agents = sorted([ua for ua in user_agents if ua != 'NOT_FOUND'])
        if unique_agents:
            ua_str = json.dumps(unique_agents)
            detailed_activity += f" User-Agent(s) observed: {ua_str}."
            if any('friendly-scanner' in ua for ua in unique_agents): detailed_activity += " (Known scanner UA detected)."
        if unique_numbers:
            pattern_hint = ""
            if len(unique_numbers) > 2:
                if all(len(n) < 5 and n.isdigit() for n in unique_numbers): pattern_hint = " (Possible extension enumeration)."
                try:
                    numeric_numbers = sorted([int(n) for n in unique_numbers if n.isdigit()])
                    if len(numeric_numbers) > 2 and (numeric_numbers[-1] - numeric_numbers[0]) == (len(numeric_numbers) - 1):
                        pattern_hint = " (Sequential numbers detected, indicates dial plan scanning)."
                except ValueError: pass
            num_examples = unique_numbers[:5]
            detailed_activity += f" Target numbers/extensions observed (examples): {json.dumps(num_examples)}{'...' if len(unique_numbers) > 5 else ''}{pattern_hint}"
        if detailed_activity:
            desc_parts.append(f"Specific observations:{detailed_activity}")

    elif honeypot_type == 'Ciscoasa':
        session_entries = log_entry_or_session if isinstance(log_entry_or_session, list) else [log_entry_or_session]
        desc_parts.append("Interaction detected with an emulated Cisco ASA device.")
        requests = [f"{match.group(1)} {match.group(2)}" for entry in session_entries if (payload := str(entry.get('payload_printable', ''))) and (match := re.search(r'(GET|POST)\s+([^\s]+)\s+HTTP/[\d\.]+', payload))]
        unique_requests = list(dict.fromkeys(requests))
        if unique_requests:
            desc_parts.append(f"{len(unique_requests)} unique HTTP request(s) were observed.")
            detailed_activity = f"HTTP Request Sequence: {json.dumps(unique_requests)}"
        else:
            first_payload = session_entries[0].get('payload_printable', 'No specific data recorded.')
            detailed_activity = f"No parsable HTTP requests found. First logged data snippet: '{first_payload[:150]}...'"
        desc_parts.append(f"Specific observations:{detailed_activity}")
    else:
        return f"Unprocessed event type '{honeypot_type}'."
    return " ".join(desc_parts)

# ==============================================================================
# 4. DEFINIZIONE DELLE CLASSI PRINCIPALI (PARSING PROFONDO & BOOST)
# ==============================================================================

class CAPECPattern:
    def __init__(self, pattern_xml: ET.Element):
        self.id = pattern_xml.get('ID')
        self.name = pattern_xml.get('Name')
        self.abstraction = pattern_xml.get('Abstraction') # Meta, Standard, Detailed
        self.status = pattern_xml.get('Status')
        
        # 1. Base Descriptions
        self.description = self._parse_block(pattern_xml, 'capec:Description')
        self.extended_description = self._parse_block(pattern_xml, 'capec:Extended_Description')
        
        # 2. Execution Flow (Techniques) - Deep Parsing
        self.techniques = []
        flow = pattern_xml.find('capec:Execution_Flow', NS)
        if flow is not None:
            for step in flow.findall('.//capec:Attack_Step', NS):
                step_desc = step.find('capec:Description', NS)
                if step_desc is not None: self.techniques.append(self._clean_text(step_desc))
                for tech in step.findall('capec:Technique', NS):
                    self.techniques.append(self._clean_text(tech))

        # 3. Examples (Critical for BM25)
        self.examples = []
        examples_block = pattern_xml.find('capec:Example_Instances', NS)
        if examples_block is not None:
            for ex in examples_block.findall('capec:Example', NS):
                self.examples.append(self._clean_text(ex))
        
        # 4. Alternate Terms
        self.alternate_terms = []
        alt_block = pattern_xml.find('capec:Alternate_Terms', NS)
        if alt_block is not None:
            for term in alt_block.findall('capec:Alternate_Term/capec:Term', NS):
                if term.text: self.alternate_terms.append(term.text.strip())

        # 5. Taxonomy Mappings (OWASP, ATT&CK, WASC)
        self.taxonomy = []
        tax_block = pattern_xml.find('capec:Taxonomy_Mappings', NS)
        if tax_block is not None:
            for tax in tax_block.findall('capec:Taxonomy_Mapping', NS):
                entry_name = tax.find('capec:Entry_Name', NS)
                if entry_name is not None and entry_name.text:
                    self.taxonomy.append(entry_name.text.strip())

        # 6. Prerequisites (Per Cross-Encoder)
        self.prerequisites = []
        prereq_block = pattern_xml.find('capec:Prerequisites', NS)
        if prereq_block is not None:
            for pre in prereq_block.findall('capec:Prerequisite', NS):
                self.prerequisites.append(self._clean_text(pre))

        # 7. Indicators (Per Cross-Encoder)
        self.indicators = []
        ind_block = pattern_xml.find('capec:Indicators', NS)
        if ind_block is not None:
            for ind in ind_block.findall('capec:Indicator', NS):
                self.indicators.append(self._clean_text(ind))

        # 8. Mitigations & CWE (Per Output Finale)
        self.mitigations = []
        mit_block = pattern_xml.find('capec:Mitigations', NS)
        if mit_block is not None:
            for mit in mit_block.findall('capec:Mitigation', NS):
                self.mitigations.append(self._clean_text(mit))
        
        self.related_weaknesses = []
        rel_weak_block = pattern_xml.find('capec:Related_Weaknesses', NS)
        if rel_weak_block is not None:
            for weak in rel_weak_block.findall('capec:Related_Weakness', NS):
                cwe_id = weak.get('CWE_ID')
                if cwe_id: self.related_weaknesses.append(f"CWE-{cwe_id}")

    def _clean_text(self, element: Optional[ET.Element]) -> str:
        if element is None: return ""
        text_content = []
        for text in element.itertext():
            if text: text_content.append(text.strip())
        return full_preprocess(" ".join(text_content))

    def _parse_block(self, pattern, path):
        elem = pattern.find(path, NS)
        return self._clean_text(elem)

    def get_bm25_text(self) -> str:
        parts = [
            self.name, self.name, self.name,
            " ".join(self.alternate_terms), " ".join(self.alternate_terms), " ".join(self.alternate_terms),
            " ".join(self.examples), " ".join(self.examples),
            " ".join(self.taxonomy), " ".join(self.taxonomy),
            self.description,
            " ".join(self.techniques)
        ]
        return " ".join(filter(None, parts))

    def get_semantic_text(self) -> str:
        parts = [
            f"Attack Pattern Name: {self.name}",
            f"Abstraction Level: {self.abstraction}",
            f"Description: {self.description}",
            f"Extended Details: {self.extended_description}",
            f"Techniques and Steps: {'; '.join(self.techniques)}",
            f"Real World Examples: {'; '.join(self.examples)}",
            f"Indicators: {'; '.join(self.indicators)}"
        ]
        return " ".join(filter(None, parts))

    def to_metadata(self) -> Dict[str, Any]:
        return {
            'id': str(self.id),
            'name': str(self.name),
            'abstraction': str(self.abstraction),
            'description': self.description[:500],
            'prerequisites': " ".join(self.prerequisites),
            'indicators': " ".join(self.indicators),
            'mitigations': " | ".join(self.mitigations[:3]),
            'related_cwe': ", ".join(self.related_weaknesses)
        }

class VectorDBManager:
    def __init__(self):
        print(f"\033[1;33m[DB] Inizializzazione Embedding SOTA ({EMBEDDING_MODEL})...\033[0m")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        
        if os.path.exists(CHROMA_DB_PATH):
            print(f"[DB] Pulizia directory database {CHROMA_DB_PATH}...")
            try:
                shutil.rmtree(CHROMA_DB_PATH)
            except Exception as e:
                print(f"[WARNING] Impossibile cancellare DB: {e}")

        print(f"\033[1;33m[DB] Connessione a ChromaDB ({CHROMA_DB_PATH})...\033[0m")
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH, settings=Settings(anonymized_telemetry=False, allow_reset=True))
        self.collection = None
        self.bm25 = None
        self.capec_ids = []
        self.capec_metadata_map = {}

    def initialize_db(self, patterns: List[CAPECPattern]):
        print(f"\033[1;36m{'=' * 80}\033[0m")
        print(f"\033[1;33m[DB] Inizializzazione del database vettoriale + BM25...\033[0m")
        
        collection_name = "capec_bge_m3_final_v5"
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

        if self.collection.count() == 0:
            print("[DB] Generazione contesti Ottimizzati (Semantic & Keyword)...")
            semantic_contexts = [p.get_semantic_text() for p in patterns]
            bm25_contexts = [p.get_bm25_text() for p in patterns]
            
            self.capec_ids = [p.id for p in patterns]
            metadatas = [p.to_metadata() for p in patterns]
            self.capec_metadata_map = {p.id: meta for p, meta in zip(patterns, metadatas)}

            print("[DB] Costruzione indice BM25...")
            tokenized_corpus = [get_enhanced_tokens(ctx) for ctx in bm25_contexts]
            self.bm25 = BM25Okapi(tokenized_corpus)

            print(f"[DB] Calcolo embeddings densi con {EMBEDDING_MODEL}...")
            all_embeddings = self.embedder.encode(semantic_contexts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)
            
            db_batch_size = 200
            for i in tqdm(range(0, len(self.capec_ids), db_batch_size), desc="[DB] Aggiunta a Chroma"):
                self.collection.add(
                    ids=self.capec_ids[i:i + db_batch_size],
                    embeddings=[e.tolist() for e in all_embeddings[i:i + db_batch_size]],
                    documents=semantic_contexts[i:i + db_batch_size], 
                    metadatas=metadatas[i:i + db_batch_size]
                )
        else:
            print("[DB] Database esistente trovato. Caricamento indici...")
            data = self.collection.get()
            contexts = data['documents']
            self.capec_ids = data['ids']
            metadatas = data['metadatas']
            self.capec_metadata_map = {id_: meta for id_, meta in zip(self.capec_ids, metadatas)}
            
            print("[DB] Ricostruzione BM25...")
            tokenized_corpus = [get_enhanced_tokens(ctx) for ctx in contexts]
            self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"\033[1;32m[DB] Database inizializzato! ({self.collection.count()} elementi)\033[0m")
        print(f"\033[1;36m{'=' * 80}\033[0m")

    def compute_bm25_scores(self, query: str) -> Dict[str, float]:
        tokenized_query = get_enhanced_tokens(query)
        if not tokenized_query: return {}
        
        scores = self.bm25.get_scores(tokenized_query)
        max_score = np.max(scores) if len(scores) > 0 else 0
        if max_score > 0:
            scores = scores / max_score
            
        return {self.capec_ids[i]: float(scores[i]) for i in range(len(self.capec_ids))}

# ==============================================================================
# 5. HYBRID ANALYZER (RRF + CROSS-ENCODER + ABSTRACTION BOOST)
# ==============================================================================
class HybridAnalyzer:
    def __init__(self, k_rrf: int = 60):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        print(f"\033[1;33m[ANALYZER] Inizializzazione Cross-Encoder ({CROSS_ENCODER_MODEL})...\033[0m")
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        self.k_rrf = k_rrf

    def _get_query_text_from_llm(self, llm_analysis: Dict) -> str:
        concept_signal = f"{llm_analysis.get('pattern_name', '')} {llm_analysis.get('summary_description', '')} {llm_analysis.get('extended_description', '')}"
        action_signal = f"{llm_analysis.get('technical_actions', '')} {llm_analysis.get('payload_sample', '')}"
        context_signal = f"{llm_analysis.get('prerequisites', '')}"
        keywords = " ".join(llm_analysis.get('technical_keywords', []))
        
        full_query = f"{concept_signal} {action_signal} {context_signal} {keywords}"
        return full_preprocess(full_query)

    def analyze(self, llm_input_description: str, honeypot_type: str, db_manager: VectorDBManager, top_k: int = 5) -> Dict[str, Any]:
        llm_analysis = _generate_llm_analysis(llm_input_description, honeypot_type)
        query_text = self._get_query_text_from_llm(llm_analysis)

        result_dict = {
            'llm_input_description': llm_input_description,
            'llm_analysis': llm_analysis,
            'db_matches': [],
            'rrf_candidates': [] # Nuova chiave per salvare lo stato pre-rerank
        }

        if not query_text: return result_dict

        # --- FASE 1: RETRIEVAL IBRIDO ---
        CANDIDATE_COUNT = 100
        query_embedding = self.embedder.encode(query_text, normalize_embeddings=True)
        semantic_results = db_manager.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=CANDIDATE_COUNT
        )
        semantic_ids = semantic_results.get('ids', [[]])[0]
        semantic_rank_map = {doc_id: i + 1 for i, doc_id in enumerate(semantic_ids)}

        bm25_scores = db_manager.compute_bm25_scores(query_text)
        sorted_bm25 = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[:CANDIDATE_COUNT]
        bm25_ids = [x[0] for x in sorted_bm25]
        bm25_rank_map = {doc_id: i + 1 for i, doc_id in enumerate(bm25_ids)}

        all_candidate_ids = set(semantic_ids) | set(bm25_ids)
        if not all_candidate_ids: return result_dict

        # --- FASE 2: RRF CON ABSTRACTION BOOST ---
        fused_scores = defaultdict(float)
        for doc_id in all_candidate_ids:
            rrf_val = 0
            if doc_id in semantic_rank_map:
                rrf_val += 1 / (self.k_rrf + semantic_rank_map[doc_id])
            if doc_id in bm25_rank_map:
                rrf_val += 1 / (self.k_rrf + bm25_rank_map[doc_id])
            
            meta = db_manager.capec_metadata_map.get(doc_id, {})
            abstraction = meta.get('abstraction', 'Standard')
            boost = 1.0
            if abstraction == 'Detailed': boost = 1.2
            elif abstraction == 'Meta': boost = 0.8
            
            fused_scores[doc_id] = rrf_val * boost

        sorted_fused_ids = sorted(fused_scores.keys(), key=lambda doc_id: fused_scores[doc_id], reverse=True)

        # Creazione lista candidati
        TOP_N_RERANK = 30
        candidates_list = []
        
        for capec_id in sorted_fused_ids[:TOP_N_RERANK]:
             if (metadata := db_manager.capec_metadata_map.get(capec_id)):
                candidate_data = {
                    'capec_id': capec_id,
                    'pattern': metadata['name'],
                    'confidence': fused_scores[capec_id], # Qui è ancora score RRF
                    'semantic_rank': semantic_rank_map.get(capec_id, '-'),
                    'bm25_rank': bm25_rank_map.get(capec_id, '-'),
                    'metadata': metadata,
                    'rrf_score': fused_scores[capec_id],
                    'cross_score': 0.0 # Placeholder
                }
                candidates_list.append(candidate_data)
        
        # SALVIAMO LO STATO RRF ORA (Copia profonda non necessaria, basta una nuova lista)
        # Questo serve per il debug: vediamo l'ordine PRIMA che il Cross-Encoder lo tocchi
        result_dict['rrf_candidates'] = list(candidates_list)

        # --- FASE 3: RE-RANKING CON CROSS-ENCODER ---
        if candidates_list:
            pairs = []
            for cand in candidates_list:
                meta = cand['metadata']
                # Costruiamo un "Documento Ricco" per il Cross-Encoder per il confronto
                doc_rich_text = (
                    f"Pattern: {cand['pattern']}. "
                    f"Description: {meta['description']} "
                    f"Prerequisites: {meta.get('prerequisites', '')} "
                    f"Indicators: {meta.get('indicators', '')}"
                )
                pairs.append([query_text, doc_rich_text])
            
            cross_scores = self.cross_encoder.predict(pairs)
            
            for i, cand in enumerate(candidates_list):
                cand['cross_score'] = float(cross_scores[i])
                # Aggiorniamo la confidence principale col Cross-Score
                cand['confidence'] = float(cross_scores[i])
            
            # Riordiniamo la lista principale (db_matches) basandoci sul Cross-Score
            candidates_list.sort(key=lambda x: x['cross_score'], reverse=True)

        result_dict['db_matches'] = candidates_list[:top_k]

        return result_dict

# ==============================================================================
# 6. LOGICA DI INTERAZIONE CON LLM (LLAMA 3.1 NATIVE PROMPTING)
# ==============================================================================

tokenizer = None
llm_pipeline = None

def _generate_llm_analysis(event_input_description: str, honeypot_type: str) -> Dict[str, Any]:
    global llm_pipeline, tokenizer
    if not llm_pipeline or not tokenizer:
        return {"pattern_name": "N/A", "detailed_description": "LLM Not Initialized", "technical_keywords": [], "justification": ""}

    # 1. Regole Contestuali (Dal tuo VECCHIO prompt, preservando la logica specifica)
    context_instructions = ""
    if honeypot_type == 'Cowrie':
        context_instructions = """
### CONTEXT: Cowrie (SSH/Telnet Honeypot)
- **Execution Rule:** If `wget`, `curl`, `chmod` or specific binaries (`./bot`) are seen, this is EXECUTION (Highest Priority).
- **Recon Rule:** If only commands like `uname`, `cat /proc/cpuinfo` are seen, this is RECONNAISSANCE.
- **Login Rule:** Failed logins with non-human usernames (e.g., 'GET /') are FINGERPRINTING, not Brute Force."""
    
    elif honeypot_type == 'Honeytrap':
        context_instructions = """
### CONTEXT: Honeytrap (Network Listener)
- **Payload Rule:** If specific hex payloads are decoded into shell commands, this is EXECUTION.
- **Exploit Rule:** Known exploit signatures (e.g., EternalBlue hex patterns) are EXPLOITATION.
- **Scanning Rule:** Empty connections or generic TLS handshakes are DISCOVERY."""

    elif honeypot_type == 'Dionaea':
        context_instructions = """
### CONTEXT: Dionaea (Malware Capture)
- **Credential Rule:** Capturing 'sa' (MSSQL) or 'root' (MySQL) credentials is CREDENTIAL ACCESS.
- **Malware Rule:** If a binary is uploaded (SMB/FTP), this is DELIVERY/EXECUTION.
- **Scanning Rule:** Connection attempts without payloads are DISCOVERY."""

    elif honeypot_type == 'Sentrypeer':
        context_instructions = """
### CONTEXT: Sentrypeer (VoIP/SIP)
- **Hijacking Rule:** `REGISTER` requests indicate SIP Account Hijacking (High Severity).
- **Enumeration Rule:** Sequential `INVITE` attempts indicate Extension Enumeration.
- **Fraud Rule:** `INVITE` to international numbers indicates Toll Fraud."""

    elif honeypot_type == 'Ciscoasa':
        context_instructions = """
### CONTEXT: CiscoASA (Web Simulation)
- **Web Recon Rule:** Requests to `/+CSCOE+/` or `/remote/logincheck` are SPECIFIC SERVICE DISCOVERY.
- **No Brute Force:** Do not classify single GET requests as Brute Force."""

    # 2. SYSTEM PROMPT IBRIDO (Logica Vecchia + Struttura Nuova)
    system_prompt = f"""You are a Threat Intelligence Expert tasked with mapping raw logs to the MITRE CAPEC framework.

### CORE MISSION: IDENTIFY THE APEX THREAT
You must analyze the log and identify the **single most severe intent**.
**Threat Hierarchy (Highest to Lowest):**
1. **Execution/Exploitation:** Code execution, malware download, command injection.
2. **Credential Access:** Successful logins, credential dumping, stealing tokens.
3. **Reconnaissance:** Scanning, fingerprinting, failed login attempts.

### ANALYSIS PROCESS (Chain of Thought)
1. **Ingest:** Read the raw log data and context.
2. **Validate:** Is the input valid for the protocol? (e.g. 'GET /' as a username is not a user, it's a protocol mismatch scan).
3. **Prioritize:** If you see both Scanning and Execution, classify as **EXECUTION**.
4. **Abstract:** Define the technique using formal MITRE language (e.g. "Adversary", "Target"), not conversational language.

{context_instructions}

### ANTI-POISONING RULES
- **NO Hallucinations:** Do not infer attacks not present in the log (e.g. don't say "SQL Injection" if it's an OS shell command).
- **NO Future/Past:** Describe only the mechanism visible in the log, not what *might* happen next.

### OUTPUT FORMAT
Provide a single JSON object.
{{
  "_reasoning": "Explain your logic here. Why did you choose this Apex Threat? What did you discard?",
  "pattern_name": "CAPEC-Style Title (e.g. 'OS Command Injection')",
  "summary_description": "A concise, 1-sentence abstract definition of the attack pattern.",
  "extended_description": "A detailed, encyclopedic description of how this technique works generally (not just this log). Use formal tone.",
  "technical_actions": "List specific technical actions found in the log (e.g. 'Inject command delimiters', 'Use wget').",
  "payload_sample": "Extract the specific command, hex snippet, or filename from the log.",
  "prerequisites": "What state must the system be in for this attack to work?",
  "technical_keywords": ["keyword1", "keyword2", "keyword3"],
  "justification": "One sentence citing the specific evidence from the log."
}}
"""

    # 3. USER PROMPT
    user_prompt = f"""Analyze the following log entry:

<log_data>
{event_input_description}
</log_data>
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = llm_pipeline(
            prompt_formatted, 
            max_new_tokens=1000, 
            temperature=0.1,    
            top_p=0.9, 
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = outputs[0]['generated_text'][len(prompt_formatted):].strip()
        return _parse_llm_response(response)
    except Exception as e:
        print(f"\033[1;31m[ERRORE] LLM: {e}\033[0m")
        return {}

def _parse_llm_response(response: str) -> Dict[str, Any]:
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start == -1 or end == 0: return {}
        
        json_str = response[start:end]
        data = json.loads(json_str)
        
        return {
            'pattern_name': data.get('pattern_name', 'Unknown'),
            'summary_description': data.get('summary_description', ''),
            'detailed_description': data.get('abstract_definition', data.get('extended_description', '')),
            'extended_description': data.get('extended_description', ''),
            'technical_actions': data.get('technical_actions', ''),
            'payload_sample': data.get('payload_sample', ''),
            'prerequisites': data.get('prerequisites', ''),
            'technical_keywords': data.get('technical_keywords', []),
            'justification': data.get('justification', '')
        }
    except Exception:
        return {}

# ==============================================================================
# 7. FUNZIONI DI INIZIALIZZAZIONE E STAMPA AGGIORNATE
# ==============================================================================

def initialize_system():
    global tokenizer, llm_pipeline
    if not os.path.exists(CAPEC_XML_FILE):
        print(f"\033[1;31m[ERRORE] File CAPEC '{CAPEC_XML_FILE}' non trovato.\033[0m"); exit()

    print(f"\033[1;44m--- INIZIO CONFIGURAZIONE SISTEMA --- \033[0m")
    try:
        tree = ET.parse(CAPEC_XML_FILE)
        patterns = [CAPECPattern(e) for e in tree.findall('.//capec:Attack_Pattern', NS)]
        print(f"\033[1;32m[CONFIG] {len(patterns)} pattern CAPEC caricati.\033[0m")
    except ET.ParseError as e: print(f"\033[1;31m[ERRORE] Parsing CAPEC XML fallito: {e}\033[0m"); exit()

    print(f"\033[1;33m[CONFIG] Configurazione LLM: {LLM_MODEL}...\033[0m")
    try:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
        llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        print(f"\033[1;32m[CONFIG] Pipeline LLM creata con successo.\033[0m")
    except Exception as e: print(f"\033[1;31m[ERRORE] Init LLM fallita: {e}\033[0m. Prova `huggingface-cli login`."); exit()

    db_manager = VectorDBManager()
    db_manager.initialize_db(patterns)
    analyzer = HybridAnalyzer(k_rrf=60)
    print(f"\033[1;42m--- CONFIGURAZIONE COMPLETATA --- \033[0m")
    return analyzer, db_manager

def print_analysis_result(result: Dict, honeypot_type: str, identifier: str, top_k: int):
    src_ip = result.get('src_ip', 'N/A')
    llm = result.get('llm_analysis', {})

    print(f"\n\033[1;44m{'='*20} 🛡️  ANALISI SESSIONE: {identifier} ({honeypot_type.upper()}) {'='*20}\033[0m")
    print(f"\033[1;37m📡 Sorgente Attacco (SRC IP):\033[0m \033[1;33m{src_ip}\033[0m")
    
    # 1. Input Originale
    print(f"\n\033[1;35m📜 Input Fornito all'LLM:\033[0m")
    print(f"\033[37m{result.get('llm_input_description', 'N/A')}\033[0m")
    
    # 2. Output LLM Dettagliato
    print("\n\033[1;36m🧠 --- INTELLIGENZA ARTIFICIALE (LLAMA 3.1 OUTPUT) ---\033[0m")
    print(f"  \033[1m🎯 Pattern Name:\033[0m \033[1;32m{llm.get('pattern_name', 'N/A')}\033[0m")
    print(f"  \033[1m📝 Summary Desc:\033[0m {llm.get('summary_description', 'N/A')}")
    print(f"  \033[1m🔎 Extended Desc:\033[0m {llm.get('extended_description', 'N/A')}")
    print(f"  \033[1m⚙️  Tech Actions:\033[0m {llm.get('technical_actions', 'N/A')}")
    print(f"  \033[1m💣 Payload:\033[0m \033[31m{llm.get('payload_sample', 'N/A')}\033[0m")
    print(f"  \033[1m🔓 Prerequisites:\033[0m {llm.get('prerequisites', 'N/A')}")
    print(f"  \033[1m🔑 Keywords:\033[0m {', '.join(llm.get('technical_keywords', []))}")
    print(f"  \033[3m⚖️  Justification:\033[0m \"{llm.get('justification', 'N/A')}\"")

    # 3. Classifica RRF (Intermedia)
    print("\n\033[1;36m🧮 --- CLASSIFICA INTERMEDIA (RRF - Reciprocal Rank Fusion) ---\033[0m")
    debug_cands = result.get('rrf_candidates', [])
    if debug_cands:
        print(f"\033[1m{'Rank':<5} | {'CAPEC ID':<10} | {'Nome Pattern':<50} | {'Sem. #':<8} | {'Key. #':<8} | {'Score RRF':<10}\033[0m")
        print("-" * 105)
        for i, c in enumerate(debug_cands[:10]): # Mostra top 10 RRF
            c_name = c['metadata']['name'][:48] + ".." if len(c['metadata']['name']) > 48 else c['metadata']['name']
            print(f"{i+1:<5} | CAPEC-{c['capec_id']:<6} | {c_name:<50} | #{c['semantic_rank']:<6} | #{c['bm25_rank']:<6} | {c['rrf_score']:.4f}")
    else:
        print("  (Nessun candidato RRF trovato)")

    # 4. Classifica Finale (Cross-Encoder)
    print("\n\033[1;32m🏆 --- CLASSIFICA FINALE (Cross-Encoder Re-Ranked) ---\033[0m")
    db_matches = result.get('db_matches', [])
    if not db_matches:
        print("  \033[1;31m❌ Nessun pattern CAPEC correlato trovato.\033[0m")
    else:
        for i, match in enumerate(db_matches):
            meta = match.get('metadata', {})
            score = match.get('cross_score', 0.0)
            
            if score > 2.0: conf_col = "\033[1;32m"
            elif score > 0.0: conf_col = "\033[1;33m"
            else: conf_col = "\033[0;37m"

            print(f"\n  \033[1m{i+1}. [{conf_col}CAPEC-{match['capec_id']}\033[0m]: \033[1;37m{match['pattern']}\033[0m")
            print(f"     📊 \033[90mCross-Score: {score:.4f} | Abstraction: {meta.get('abstraction', '-')}\033[0m")
            print(f"     📍 \033[90mPosizione negli indici -> Semantico: #{match.get('semantic_rank','-')} | Keyword(BM25): #{match.get('bm25_rank','-')}\033[0m")
            
            # Descrizione breve
            desc = meta.get('description', '')
            if len(desc) > 200: desc = desc[:200] + "..."
            print(f"     📖 {desc}")

    print("\033[1;44m" + "="*80 + "\033[0m\n")

# ==============================================================================
# 8. BLOCCO DI ESECUZIONE PRINCIPALE
# ==============================================================================

if __name__ == "__main__":
    TOP_K_RESULTS = 5 

    if not os.path.exists(HONEYPOT_CSV_FILE):
        print(f"\033[1;31m[ERRORE] File CSV '{HONEYPOT_CSV_FILE}' non trovato.\033[0m"); exit()

    analyzer, db_manager = initialize_system()

    print(f"\n\033[1;44m--- INIZIO ELABORAZIONE DATI --- \033[0m")
    try:
        df = pd.read_csv(HONEYPOT_CSV_FILE, delimiter=';', encoding='utf-8-sig', 
                         dtype={'called_number': str, 'dest_port': 'Int64'}, low_memory=False)
        print(f"\033[1;32m[DATI] Caricate {len(df)} righe totali.\033[0m")

        if '@timestamp' in df.columns:
            if 'timestamp' in df.columns: df.drop(columns=['timestamp'], inplace=True)
            df.rename(columns={'@timestamp': 'timestamp'}, inplace=True)
        
        if 'timestamp' not in df.columns:
             print(f"\033[1;31m[ERRORE] Colonna 'timestamp' non trovata.\033[0m"); exit()
        
        allowed_types = ['Cowrie', 'Honeytrap', 'Dionaea', 'Sentrypeer', 'Ciscoasa']
        if 'type' not in df.columns:
             print(f"\033[1;31m[ERRORE] Colonna 'type' mancante.\033[0m"); exit()
             
        df_filtered = df[df['type'].isin(allowed_types)].copy()
        
        noise_patterns = ['Traceback', 'Exception occurred', 'Stopping server', 'Request timed out', 'ssl.SSLEOFError', 'socketserver.py', 'NameError', 'RecursionError']
        if 'message' in df_filtered.columns:
             for pattern in noise_patterns:
                df_filtered = df_filtered[~df_filtered['message'].str.contains(pattern, na=False)]

        df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'], errors='coerce')
        
        if 'dest_port' in df_filtered.columns:
             df_filtered['dest_port'] = df_filtered['dest_port'].fillna(0).astype(int)

    except Exception as e:
        print(f"\033[1;31m[ERRORE] Lettura/Elaborazione CSV: {e}\033[0m"); traceback.print_exc(); exit()

    base_required_cols = ['type', 'timestamp', 'src_ip']
    df_filtered.dropna(subset=base_required_cols, inplace=True)
    
    if len(df_filtered) == 0: 
        print(f"\033[1;31m[ERRORE] Nessuna riga valida dopo il filtro.\033[0m"); exit()

    sessions_to_process = defaultdict(list)

    if 'session' in df_filtered.columns:
        cowrie_df = df_filtered[(df_filtered['type'] == 'Cowrie') & df_filtered['session'].notna()].copy()
        for session_id, group in cowrie_df.groupby('session'):
            sessions_to_process[session_id] = group.sort_values(by='timestamp').to_dict('records')

    other_hps = ['Ciscoasa', 'Dionaea', 'Honeytrap', 'Sentrypeer']
    for ht in other_hps:
        ht_df = df_filtered[df_filtered['type'] == ht].copy()
        if not ht_df.empty:
            for src_ip, group in ht_df.groupby('src_ip'):
                session_id = f"{ht.lower()}_{src_ip}"
                sessions_to_process[session_id] = group.sort_values(by='timestamp').to_dict('records')

    print(f"\033[1;32m[RAGGRUPPAMENTO] Create {len(sessions_to_process)} sessioni totali da analizzare.\033[0m")

    analyzed_count = 0
    if sessions_to_process:
        for session_id, entries in tqdm(sessions_to_process.items(), desc="[ANALISI] Processo Sessioni"):
            h_type = entries[0]['type']
            
            try:
                llm_input = generate_llm_input(entries, h_type)
            except Exception as e:
                print(f"[WARN] Errore generazione input per {session_id}: {e}")
                continue

            try:
                result = analyzer.analyze(llm_input, h_type, db_manager, top_k=TOP_K_RESULTS)
                result['src_ip'] = entries[0].get('src_ip', 'N/A')
                print_analysis_result(result, h_type, f"Sessione: {session_id}", TOP_K_RESULTS)
                analyzed_count += 1
                
            except Exception as e:
                print(f"\033[1;31m[ERRORE] Analisi sessione {session_id} fallita: {e}\033[0m"); traceback.print_exc()

    print(f"\n\033[1;42m{'=' * 80}\033[0m")
    print(f"\033[1;32mANALISI GLOBALE COMPLETATA. {analyzed_count} eventi analizzati con successo.\033[0m")
