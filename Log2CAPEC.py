import re
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, Optional, Union
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import binascii
import traceback
from tqdm import tqdm
from collections import defaultdict

# ==============================================================================
# 1. CONFIGURAZIONE INIZIALE E COSTANTI
# ==============================================================================

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("\033[1;33m[CONFIGURAZIONE] Download del modello Spacy...\033[0m")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("\033[1;33m[CONFIGURAZIONE] Download dei dati NLTK...\033[0m")
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

NS = {'capec': 'http://capec.mitre.org/capec-3', 'xhtml': 'http://www.w3.org/1999/xhtml'}
EMBEDDING_MODEL = 'basel/ATTACK-BERT'
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
CHROMA_DB_PATH = "./chroma_db_all_hps_v3_focused"
CAPEC_XML_FILE = "CAPEC.xml"
HONEYPOT_CSV_FILE = "tpot_less.csv"

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
    if not isinstance(text, str): return []
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if
              not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN'] and len(token.lemma_) > 2]
    bigrams = [' '.join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]
    trigrams = [' '.join(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
    return [t for t in tokens + bigrams + trigrams if t and t.strip()]

def preprocess_text(text: str) -> str:
    if not isinstance(text, str): return str(text)
    text = text.lower()
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '', text)
    text = re.sub(r'\bv(?:ersion)?\s*\d+(\.\d+)+(\.\d+)*\b', '', text)
    text = re.sub(r'\b\d+(\.\d+)+\b', ' number ', text)
    text = re.sub(r'\b\d+\b', ' number ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_unwanted_terms(text: str) -> str:
    if not isinstance(text, str): return str(text)
    text = re.sub(r'\b(capec|cwe)\s*\d*\b', '', text, flags=re.IGNORECASE)
    generic_terms = ['description', 'mitigations', 'mitigation', 'execution flow', 'prerequisites', 'prerequisite',
                     'summary', 'details', 'content', 'example', 'reference', 'related', 'attack', 'pattern', 'step',
                     'phase', 'objective', 'technique', 'procedure', 'tactic', 'common', 'consequence', 'likelihood',
                     'severity', 'skill', 'resource', 'required', 'typical', 'various', 'attacker', 'adversary',
                     'system', 'target', 'victim', 'user', 'application', 'network', 'service', 'protocol', 'data',
                     'information', 'access', 'control', 'mechanism', 'security', 'ensure', 'prevent', 'detect',
                     'response', 'implement', 'strategy', 'approach', 'method', 'process', 'result', 'effect', 'impact',
                     'note', 'introduction', 'background', 'also', 'however', 'therefore']
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
# 4. DEFINIZIONE DELLE CLASSI PRINCIPALI
# ==============================================================================

class CAPECPattern:
    def __init__(self, pattern_xml: ET.Element):
        self.id = pattern_xml.get('ID')
        self.name = pattern_xml.get('Name')
        self.description = self._parse_description(pattern_xml)
        self.execution_flow = self._parse_execution_flow(pattern_xml)
        self.prerequisites = self._parse_generic_text_or_description(pattern_xml, 'capec:Prerequisites/capec:Prerequisite')
        self.mitigations = self._parse_generic_text_or_description(pattern_xml, 'capec:Mitigations/capec:Mitigation')
        self.likelihood = pattern_xml.findtext('capec:Likelihood_Of_Attack', default='Not Specified', namespaces=NS)
        self.severity = pattern_xml.findtext('capec:Typical_Severity', default='Not Specified', namespaces=NS)

    def _parse_text_from_complex_type(self, element: Optional[ET.Element]) -> str:
        texts = []
        if element is not None:
            if element.text:
                texts.append(element.text.strip())
            for p in element.findall('.//xhtml:p', NS):
                if p.text:
                    texts.append(p.text.strip())
        return full_preprocess(' '.join(filter(None, texts)))

    def _parse_description(self, pattern: ET.Element) -> str:
        desc_elem = pattern.find('capec:Description', NS)
        return self._parse_text_from_complex_type(desc_elem)

    def _parse_generic_text_or_description(self, pattern: ET.Element, path: str) -> List[str]:
        items = []
        for elem in pattern.findall(path, NS):
            desc_elem = elem.find('capec:Description', NS)
            processed_text = self._parse_text_from_complex_type(desc_elem) if desc_elem is not None else full_preprocess(elem.text or '')
            if processed_text:
                items.append(processed_text)
        return items

    def _parse_execution_flow(self, pattern: ET.Element) -> List[str]:
        steps = []
        flow_elem = pattern.find('capec:Execution_Flow', NS)
        if flow_elem is not None:
            for step in flow_elem.findall('.//capec:Attack_Step', NS):
                processed_text = self._parse_text_from_complex_type(step.find('capec:Description', NS))
                if processed_text:
                    steps.append(processed_text)
        return steps

    def semantic_context(self) -> str:
        parts = [
            f"CAPEC {self.id}: {self.name}",
            f"Description: {self.description}",
            f"Execution Flow: {'; '.join(self.execution_flow) if self.execution_flow else 'Not specified'}",
            f"Prerequisites: {'; '.join(self.prerequisites) if self.prerequisites else 'Not specified'}",
            f"Likelihood: {self.likelihood}",
            f"Severity: {self.severity}"
        ]
        return full_preprocess(' '.join(filter(None, parts)))

    def to_metadata(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else 'N/A',
            'name': str(self.name) if self.name else 'N/A',
            'description': str(self.description) if self.description else '',
            'execution_flow': '; '.join(self.execution_flow) if self.execution_flow else '',
            'prerequisites': '; '.join(self.prerequisites) if self.prerequisites else '',
            'mitigations': '; '.join(self.mitigations) if self.mitigations else '',
            'likelihood': str(self.likelihood) if self.likelihood else 'Not Specified',
            'severity': str(self.severity) if self.severity else 'Not Specified'
        }

class VectorDBManager:
    def __init__(self):
        print(f"\033[1;33m[DB] Inizializzazione SentenceTransformer ({EMBEDDING_MODEL})...\033[0m")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        print(f"\033[1;33m[DB] Connessione a ChromaDB ({CHROMA_DB_PATH})...\033[0m")
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH, settings=Settings(anonymized_telemetry=False, allow_reset=True))
        self.collection, self.tfidf_vectorizer, self.tfidf_matrix, self.capec_ids, self.capec_metadata_map = None, None, None, [], {}

    def initialize_db(self, patterns: List[CAPECPattern]):
        print(f"\033[1;36m{'=' * 80}\033[0m")
        print(f"\033[1;33m[DB] Inizializzazione del database vettoriale...\033[0m")
        try:
            collection_name = "capec_collection"
            try:
                self.client.delete_collection(name=collection_name)
                print("[DB] Collezione precedente rimossa.")
            except Exception: pass
            self.collection = self.client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
            print(f"\033[1;32m[DB] Collezione '{collection_name}' creata.\033[0m")

            contexts = [p.semantic_context() for p in patterns]
            self.capec_ids = [p.id for p in patterns]
            metadatas = [p.to_metadata() for p in patterns]
            self.capec_metadata_map = {p.id: meta for p, meta in zip(patterns, metadatas)}

            self.tfidf_vectorizer = TfidfVectorizer(tokenizer=get_enhanced_tokens, preprocessor=full_preprocess, max_df=0.85, min_df=2, ngram_range=(1, 3), stop_words='english')
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(contexts)
            print("[DB] Matrice TF-IDF calcolata.")

            all_embeddings = self.embedder.encode(contexts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
            print(f"\033[1;32m[DB] Embeddings calcolati.\033[0m")

            db_batch_size = 500
            for i in tqdm(range(0, len(self.capec_ids), db_batch_size), desc="[DB] Aggiunta a Chroma"):
                self.collection.add(
                    ids=self.capec_ids[i:i + db_batch_size],
                    embeddings=[e.tolist() for e in all_embeddings[i:i + db_batch_size]],
                    documents=contexts[i:i + db_batch_size],
                    metadatas=metadatas[i:i + db_batch_size]
                )
            print(f"\033[1;32m[DB] Database inizializzato! ({self.collection.count()} elementi)\033[0m")
        except Exception as e:
            print(f"\033[1;31m[ERRORE] Inizializzazione DB fallita: {e}\033[0m"); traceback.print_exc(); raise
        print(f"\033[1;36m{'=' * 80}\033[0m")

    def compute_keyword_similarity(self, query: str) -> Dict[str, float]:
        query_vec = self.tfidf_vectorizer.transform([full_preprocess(query)])
        cos_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        return {self.capec_ids[i]: float(cos_sim[i]) for i in range(len(self.capec_ids))}

    def query_semantic_top_k(self, query_embedding: np.ndarray, top_k: int) -> Dict[str, float]:
        results = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
        ids, distances = results.get('ids', [[]])[0], results.get('distances', [[]])[0]
        return {ids[i]: 1.0 - distances[i] for i in range(len(ids))} if ids and distances is not None else {}

# ==============================================================================
# MODIFICA: Implementazione di Reciprocal Rank Fusion (RRF)
# ==============================================================================
class HybridAnalyzer:
    def __init__(self, k_rrf: int = 60):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.k_rrf = k_rrf

    def _get_query_text_from_llm(self, llm_analysis: Dict) -> str:
        return full_preprocess(' '.join(filter(None, [
            llm_analysis.get('pattern_name', ''),
            llm_analysis.get('detailed_description', ''),
            ' '.join(llm_analysis.get('technical_keywords', []))
        ])))

    def analyze(self, llm_input_description: str, honeypot_type: str, db_manager: VectorDBManager, top_k: int = 5) -> Dict[str, Any]:
        llm_analysis = _generate_llm_analysis(llm_input_description, honeypot_type)
        query_text = self._get_query_text_from_llm(llm_analysis)

        result_dict = {
            'llm_input_description': llm_input_description,
            'llm_analysis': llm_analysis,
            'db_matches': [],
            'debug_candidates': []
        }

        if not query_text: return result_dict

        # --- FASE 1: OTTENERE LE LISTE ORDINATE SEPARATAMENTE ---
        CANDIDATE_COUNT = 100

        # Ricerca Semantica
        query_embedding = self.embedder.encode(query_text, normalize_embeddings=True)
        semantic_results = db_manager.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=CANDIDATE_COUNT
        )
        semantic_ids = semantic_results.get('ids', [[]])[0]
        semantic_rank_map = {doc_id: i + 1 for i, doc_id in enumerate(semantic_ids)}

        # Ricerca Keyword
        all_keyword_scores = db_manager.compute_keyword_similarity(query_text)
        sorted_keyword_ids = [k for k, v in sorted(all_keyword_scores.items(), key=lambda item: item[1], reverse=True)]
        keyword_rank_map = {doc_id: i + 1 for i, doc_id in enumerate(sorted_keyword_ids)}

        all_candidate_ids = set(semantic_ids) | set(sorted_keyword_ids[:CANDIDATE_COUNT])

        if not all_candidate_ids: return result_dict

        # --- FASE 2: APPLICARE RECIPROCAL RANK FUSION (RRF) ---
        fused_scores = defaultdict(float)
        for doc_id in all_candidate_ids:
            if doc_id in semantic_rank_map:
                fused_scores[doc_id] += 1 / (self.k_rrf + semantic_rank_map[doc_id])
            if doc_id in keyword_rank_map:
                fused_scores[doc_id] += 1 / (self.k_rrf + keyword_rank_map[doc_id])

        sorted_fused_ids = sorted(fused_scores.keys(), key=lambda doc_id: fused_scores[doc_id], reverse=True)

        # --- FASE 3: COSTRUIRE I RISULTATI PER L'OUTPUT E IL DEBUG ---
        debug_candidates_list = []
        for rank, capec_id in enumerate(sorted_fused_ids[:20]):
             if (metadata := db_manager.capec_metadata_map.get(capec_id)):
                candidate_data = {
                    'capec_id': capec_id,
                    'pattern': metadata['name'],
                    'confidence': fused_scores[capec_id],
                    'semantic_rank': semantic_rank_map.get(capec_id, 'N/A'),
                    'keyword_rank': keyword_rank_map.get(capec_id, 'N/A'),
                    'metadata': metadata
                }
                debug_candidates_list.append(candidate_data)

        result_dict['db_matches'] = debug_candidates_list[:top_k]
        result_dict['debug_candidates'] = debug_candidates_list

        return result_dict

# ==============================================================================
# 5. LOGICA DI INTERAZIONE CON LLM
# ==============================================================================

def _get_llm_prompt(event_input_description: str, honeypot_type: str) -> str:
    base_template = """<s>[INST]
### Your Role: Threat Intelligence Lexicographer
You are an AI tasked with defining cybersecurity attack patterns for a formal knowledge base like CAPEC. Your role is **not to narrate a specific event**, but to provide a formal, abstract, and encyclopedic **DEFINITION** of a single attack technique, demonstrating deep contextual understanding.

### Core Mission: Identify the Apex Threat and Define It
1.  **Identify the Apex Threat**: From the input log, identify the single most severe and significant action (e.g., Remote Code Execution, Application Fingerprinting, Credential Brute-Forcing). This is the *only* technique you will define.
2.  **Define the Technique**: Write a formal, general-purpose definition of that technique.

### Chain-of-Thought Process
You MUST follow this internal reasoning process for ALL tasks:
1.  **Evidence Ingestion**: List all actions and inputs explicitly mentioned in the log.
2.  **Input Validity Check**: Critically evaluate if inputs are contextually valid for their actions.
3.  **Intent Inference**: Determine the *true intent* of each action. An action labeled 'failed login' with an invalid username is not 'Credential Access', it is 'Reconnaissance'.
4.  **Apex Threat Identification**: Using the `Threat Hierarchy` and `Context-Specific Rules`, identify the single action with the most severe *true intent*.
5.  **Contextual Coherence Check**: Ask: "Is the inferred intent logically possible given the context?" (e.g., 'Privilege Escalation' is impossible if the user is already 'root'). Correct your analysis if it's illogical.
6.  **Abstraction & Definition**: Formulate a completely abstract definition of the *corrected* Apex Threat.

### Threat Hierarchy (Highest to Lowest Priority)
1.  **Execution**: Any form of code or command execution.
2.  **Credential Access**: Attempts to use, steal, or guess credentials.
3.  **Reconnaissance & Discovery**: All other information gathering activities.

### Critical Rules
-   **RULE 1 (EVIDENCE-BOUND REASONING)**: You MUST NOT infer actions not present in the log. If the log only shows `GET` requests, you cannot infer a `Brute Force` attack.
-   **RULE 2 (DO NOT SUMMARIZE)**: Your `detailed_description` MUST be a formal DEFINITION of the technique, not a summary of the log's events.
-   **RULE 3 (STRICT ABSTRACTION)**: The `detailed_description` and `technical_keywords` MUST NOT contain any specific details from the logs (commands, filenames, tools, etc.).
-   **RULE 4 (LOGICAL COHERENCE)**: Your analysis must be logically consistent. DO NOT classify an action with an intent that is already fulfilled. For example, **DO NOT classify an action as 'Privilege Escalation' if the user already has root/administrator privileges.**

### Output JSON Structure
Your output must be ONLY a single valid JSON object with these four keys:
-   `"pattern_name"`: Formal name as 'Broad Category: Specific Technique'.
-   `"detailed_description"`: The formal, abstract, textbook-style definition of the Apex Threat.
-   `"technical_keywords"`: A list of 8-10 abstract concepts related to the technique.
-   `"justification"`: The ONLY field where you must cite the specific evidence for the Apex Threat.
"""

    context_str, example_str = "", ""

    if honeypot_type == 'Cowrie':
        context_str = """
### Honeypot Context: Cowrie
-   **Input Type**: Interactive SSH/Telnet session logs.
-   **Context-Specific Rules**:
    1.  The execution of any script or binary (`sh`, `bash`, `./filename`) or the use of a download utility (`wget`, `curl`) is ALWAYS the Apex Threat (Execution).
    2.  If no execution occurs, a sequence of commands for system enumeration (`ifconfig`, `uname`, `ps`, `cat /proc/...`) is the Apex Threat (Reconnaissance).
    3.  A failed login with a non-human-readable username (e.g., an HTTP request) is `Application Fingerprinting`, NOT `Brute Force`."""
        example_str = """

### Example
**Input Log:**
`Interactive SSH session. Login succeeded as 'pi'. Commands: ["uname -r", "ifconfig", "ps aux"].`
**Output JSON:**
```json
{{
  "pattern_name": "Reconnaissance: System Information Gathering",
  "detailed_description": "System Information Gathering is a technique where an adversary executes a series of built-in commands to obtain detailed information about a host's configuration. This can include discovering the operating system version, network interfaces, and running processes. This activity allows the adversary to map the system's environment to plan subsequent actions.",
  "technical_keywords": ["reconnaissance", "discovery", "system information gathering", "os fingerprinting", "network configuration discovery", "process discovery", "host enumeration", "post-exploitation"],
  "justification": "The sequence of commands including 'uname -r', 'ifconfig', and 'ps aux' is the definitive evidence of a system information gathering technique."
}}
```"""

    elif honeypot_type == 'Honeytrap':
        context_str = """
### Honeypot Context: Honeytrap
-   **Input Type**: Network connection summary.
-   **Context-Specific Rules**:
    1.  A payload containing shell commands (`rm`, `wget`, etc.) within a URL is ALWAYS the Apex Threat (Execution: Command Injection).
    2.  If no command injection is present, a payload that is a valid HTTP request targeting a known vulnerability path (e.g., `/.env`, `/setup.cgi`) is the Apex Threat (Reconnaissance: Vulnerability Probing).
    3.  Payloads identified as TLS Handshakes or simple HTTP requests to `/` are basic `Service Discovery` (Reconnaissance)."""
        example_str = """

### Example
**Input Log:**
`Observed 10 TCP attempts. 1 had a payload. Example Payloads: ["GET /.env HTTP/1.1"]`
**Output JSON:**
```json
{{
  "pattern_name": "Reconnaissance: Sensitive File Discovery",
  "detailed_description": "Sensitive File Discovery is a reconnaissance technique where an adversary sends HTTP requests to probe for well-known configuration or environment files that may contain credentials, API keys, or other sensitive data. This automated activity targets common file paths to identify misconfigurations and gather information for further exploitation.",
  "technical_keywords": ["reconnaissance", "information gathering", "sensitive data exposure", "configuration file access", "http get", "vulnerability scanning", "web application mapping", "path traversal"],
  "justification": "The GET request targeting the '/.env' file, a common location for sensitive environment variables, is the key evidence of this technique."
}}
```"""
    
    elif honeypot_type == 'Dionaea':
        context_str = """
### Honeypot Context: Dionaea
-   **Input Type**: Emulated service interaction summary.
-   **Context-Specific Rules**:
    1.  The capture of specific credentials for a high-value service (e.g., 'sa' for mssql, 'root' for mysql) is ALWAYS the Apex Threat (Credential Access).
    2.  An FTP command sequence that involves more than just login (e.g., `LIST`, `STOR`, `RETR`) is the next most severe threat (Reconnaissance or Execution).
    3.  Simple connection attempts across multiple ports are `Service Discovery` (Reconnaissance)."""
        example_str = """

### Example
**Input Log:**
`Observed 15 connection attempts targeting 'mssqld' and 'smbd'. Credentials were captured. Summary: ["service: mssqld, user: 'sa'"].`
**Output JSON:**
```json
{{
  "pattern_name": "Credential Access: Database Authentication Probing",
  "detailed_description": "Database Authentication Probing is a form of credential stuffing where an adversary attempts to gain unauthorized access to a database service. The technique involves authenticating with common, default, or previously compromised administrative credentials. This action is distinct from simple service discovery, as it represents a direct attempt to compromise an account on a high-value target.",
  "technical_keywords": ["credential access", "brute-force", "service discovery", "database security", "authentication bypass", "credential stuffing", "default credentials", "reconnaissance"],
  "justification": "The capture of the 'sa' username for the mssqld service was the key evidence that elevated this activity from a simple scan to a targeted credential attack."
}}
```"""
        
    elif honeypot_type == 'Sentrypeer':
        context_str = """
### Honeypot Context: Sentrypeer
-   **Input Type**: SIP/VoIP interaction summary.
-   **Context-Specific Rules**:
    1.  A `REGISTER` request is ALWAYS the Apex Threat (Credential Access: Registration Hijacking).
    2.  A high volume of `INVITE` requests with sequential or patterned numbers is the Apex Threat (Reconnaissance: Dial Plan Enumeration).
    3.  A single `INVITE` to an international number is a potential `Toll Fraud` attempt (Execution).
    4.  `OPTIONS` requests, even with known scanner UAs, are the lowest priority threat (Reconnaissance: Endpoint Discovery)."""
        example_str = """

### Example
**Input Log:**
`Observed 2 SIP interaction(s), using method(s): REGISTER. User-Agent(s) observed: ["friendly-scanner"]. Target numbers/extensions observed: ["1000"].`
**Output JSON:**
```json
{{
  "pattern_name": "Credential Access: SIP Registration Hijacking Attempt",
  "detailed_description": "SIP Registration Hijacking is a credential access technique where an adversary attempts to register a user agent to a SIP registrar on behalf of a legitimate user. By sending crafted REGISTER requests, the attacker aims to associate their own location with the victim's SIP address, allowing them to intercept incoming calls or make fraudulent calls. This is a direct attempt to compromise an account.",
  "technical_keywords": ["credential access", "sip protocol", "voip security", "registration hijacking", "account takeover", "man-in-the-middle", "authentication bypass", "session initiation"],
  "justification": "The use of the SIP 'REGISTER' method is the key evidence, as its primary purpose is to authenticate and register an endpoint, making it an Apex Threat over simple discovery."
}}
```"""
        
    elif honeypot_type == 'Ciscoasa':
        context_str = """
### Honeypot Context: Ciscoasa
-   **Input Type**: HTTP request sequence summary.
-   **Context-Specific Rules**:
    1.  A request targeting a known, specific vulnerability or administrative path (e.g., '/+CSCOE+/', '/remote/logincheck') is ALWAYS the Apex Threat (Reconnaissance: Specific Service Discovery).
    2.  DO NOT infer a Brute Force attack from a single `GET` request to a login path. This is Reconnaissance. Brute Force can only be inferred from a high volume of `POST` requests.
    3.  Generic requests (`GET /`) are the lowest priority."""
        example_str = """

### Example
**Input Log:**
`2 unique HTTP request(s) were observed. HTTP Request Sequence: ["GET /", "GET /.git/config"]`
**Output JSON:**
```json
{{
  "pattern_name": "Reconnaissance: Sensitive File Discovery via Web Path",
  "detailed_description": "Sensitive File Discovery is a reconnaissance technique where an adversary sends HTTP requests to probe for well-known files that contain sensitive system or application data. Targeting common paths like version control system directories (e.g., '/.git/') can expose source code, configuration details, or credentials, providing valuable information for further exploitation.",
  "technical_keywords": ["reconnaissance", "information gathering", "sensitive data exposure", "version control system", "http get", "vulnerability scanning", "web application mapping", "source code disclosure"],
  "justification": "The GET request targeting the '/.git/config' file is the key evidence of an attempt to access sensitive configuration data."
}}
```"""
    
    else:
        context_str = ""
        example_str = ""
    
    final_prompt_template = (
        base_template + 
        context_str + 
        example_str +
        """

Now, analyze the following real log data, strictly adhering to all general and context-specific rules.

**Input Log:**
`{event_input_description}`

**Output JSON:**
```json
[/INST]
"""
    )
    
    return final_prompt_template.format(event_input_description=event_input_description)

tokenizer = None
llm_pipeline = None

def _generate_llm_analysis(event_input_description: str, honeypot_type: str) -> Dict[str, Any]:
    global llm_pipeline, tokenizer
    if not llm_pipeline or not tokenizer:
        return {"pattern_name": "N/A", "detailed_description": "LLM Not Initialized", "technical_keywords": [], "justification": ""}

    prompt = _get_llm_prompt(event_input_description, honeypot_type)

    try:
        response = llm_pipeline(prompt, max_new_tokens=1500, temperature=0.1, top_p=0.9, do_sample=True, repetition_penalty=1.1,
                                 pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)[0]['generated_text']
        return _parse_llm_response(response)
    except Exception as e:
        print(f"\033[1;31m[ERRORE] Chiamata pipeline LLM fallita: {e}\033[0m")
        return {"pattern_name": "N/A", "detailed_description": f"LLM Pipeline Error: {e}", "technical_keywords": [], "justification": ""}

def _parse_llm_response(response: str) -> Dict[str, Any]:
    try:
        inst_end_tag = "[/INST]"
        if inst_end_tag in response:
            response = response.split(inst_end_tag, 1)[-1].strip()

        start = response.find('{')
        end = response.rfind('}')
        if start == -1 or end == -1 or end < start:
            return {"pattern_name": "N/A", "detailed_description": f"LLM Parsing Error: No JSON block found. Response: {response[:500]}", "technical_keywords": [], "justification": ""}

        json_str = response[start:end + 1]
        parsed = json.loads(json_str)

        return {
            'pattern_name': str(parsed.get('pattern_name', 'N/A')),
            'detailed_description': str(parsed.get('detailed_description', '')),
            'technical_keywords': [str(item) for item in parsed.get('technical_keywords', []) if isinstance(parsed.get('technical_keywords'), list)],
            'justification': str(parsed.get('justification', ''))
        }
    except (json.JSONDecodeError, Exception) as e:
        return {"pattern_name": "N/A", "detailed_description": f"LLM Parsing Error: {e}. Raw response: {response[:500]}", "technical_keywords": [], "justification": ""}

# ==============================================================================
# 6. FUNZIONI DI INIZIALIZZAZIONE E STAMPA
# ==============================================================================

def initialize_system():
    global tokenizer, llm_pipeline
    if not os.path.exists(CAPEC_XML_FILE):
        print(f"\033[1;31m[ERRORE] File CAPEC '{CAPEC_XML_FILE}' non trovato.\033[0m"); exit()

    print(f"\033[1;44m--- INIZIO CONFIGURAZIONE SISTEMA --- \033[0m")

    print(f"\033[1;33m[CONFIGURAZIONE] Caricamento CAPEC da: {CAPEC_XML_FILE}\033[0m")
    try:
        tree = ET.parse(CAPEC_XML_FILE)
        patterns = [CAPECPattern(e) for e in tree.findall('.//capec:Attack_Pattern', NS)]
        print(f"\033[1;32m[CONFIGURAZIONE] {len(patterns)} pattern CAPEC caricati.\033[0m")
    except ET.ParseError as e:
        print(f"\033[1;31m[ERRORE] Parsing CAPEC XML fallito: {e}\033[0m"); exit()

    print(f"\033[1;33m[CONFIGURAZIONE] Configurazione LLM locale: {LLM_MODEL}...\033[0m")
    try:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}
        print(f"[CONFIGURAZIONE] Uso della device map: {device_map}")

        model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, quantization_config=bnb_config, device_map=device_map, torch_dtype=torch.bfloat16, trust_remote_code=True)
        llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        print(f"\033[1;32m[CONFIGURAZIONE] Pipeline LLM creata con successo.\033[0m")
    except Exception as e:
        print(f"\033[1;31m[ERRORE] Inizializzazione LLM locale fallita: {e}\033[0m")
        print("\033[1;33m[SUGGERIMENTO] Assicurati che 'torch', 'transformers', 'accelerate', 'bitsandbytes', 'scipy' siano installati e che ci sia abbastanza VRAM.\033[0m"); exit()

    db_manager = VectorDBManager()
    db_manager.initialize_db(patterns)
    analyzer = HybridAnalyzer(k_rrf=60)

    print(f"\033[1;42m--- CONFIGURAZIONE COMPLETATA --- \033[0m")
    return analyzer, db_manager

def print_analysis_result(result: Dict, honeypot_type: str, identifier: str, top_k: int):
    src_ip = result.get('src_ip', 'N/A')

    print(f"\n\033[1;44m{'='*15} 🔍 ANALISI PER: {identifier} ({honeypot_type.upper()}) {'='*15}\033[0m")
    print(f"\033[1;34mSorgente Attacco (SRC IP):\033[0m \033[1;37m{src_ip}\033[0m")
    print("\033[90m" + "-"*80 + "\033[0m")

    print("\033[1;35m📜 Input Fornito all'LLM:\033[0m")
    print(f"   \033[37m{result.get('llm_input_description', 'N/A')}\033[0m")
    print("\033[90m" + "-"*80 + "\033[0m")

    llm_analysis = result.get('llm_analysis', {})
    print("\033[1;35m📝 Risultato Analisi LLM:\033[0m")
    print(f"  \033[1mPattern Inferito:\033[0m \033[1;37m{llm_analysis.get('pattern_name', 'N/A')}\033[0m")
    if justification := llm_analysis.get('justification'):
      print(f"  \033[1mGiustificazione LLM:\033[0m \033[3m\033[90m\"{justification}\"\033[0m")
    print(f"  \033[1mDescrizione Dettagliata:\033[0m")
    print(f"  \033[37m{llm_analysis.get('detailed_description', 'N/A')}\033[0m")
    keywords = llm_analysis.get('technical_keywords', [])
    if keywords: print(f"  \033[1mKeywords Tecniche:\033[0m \033[37m{', '.join(keywords)}\033[0m")
    print("\033[90m" + "-"*80 + "\033[0m")

    print("\033[1;35m🔗 Correlazione con Pattern di Attacco Noti (CAPEC - Top {k}):\033[0m".format(k=top_k))
    db_matches = result.get('db_matches', [])
    if not db_matches:
        print("  \033[1;31mNessun pattern CAPEC correlato trovato con una confidenza sufficiente.\033[0m")
    else:
        max_score = db_matches[0]['confidence'] if db_matches else 1.0

        for i, match in enumerate(db_matches):
            relative_confidence = (match.get('confidence', 0.0) / max_score) if max_score > 0 else 0.0

            if relative_confidence > 0.90: confidence_color = "\033[1;32m"
            elif relative_confidence > 0.75: confidence_color = "\033[1;33m"
            else: confidence_color = "\033[0;37m"

            print(f"\n  \033[1m({i + 1}) CAPEC-{match.get('capec_id', 'N/A')}:\033[0m \033[1;37m{match.get('pattern', 'N/A')}\033[0m")
            print(f"     {confidence_color}Confidenza Relativa: {relative_confidence:.2%}\033[0m")
            print(f"     \033[90mDettaglio Ranghi -> Semantico: #{match.get('semantic_rank', 'N/A')}, Keyword: #{match.get('keyword_rank', 'N/A')}\033[0m")

            desc = match.get('metadata', {}).get('description', 'N/A')
            print(f"     \033[1mDescrizione CAPEC:\033[0m \033[37m{desc[:200]}...\033[0m")

    debug_candidates = result.get('debug_candidates', [])
    if debug_candidates:
        print("\033[90m" + "-"*80 + "\033[0m")
        print("\033[1;36m📊 TABELLA DI DEBUG DELLA FUSIONE RRF (Top 20 Candidati):\033[0m")
        header = f"\033[1m{'Rank':<5} | {'CAPEC ID':<10} | {'Rango Sem.':<12} | {'Rango Key.':<12} | {'Punteggio RRF':<15}\033[0m"
        print(header)
        print("\033[90m" + "-" * 70 + "\033[0m")

        for i, candidate in enumerate(debug_candidates):
            line = f"{i + 1:<5} | CAPEC-{candidate['capec_id']:<9} | #{str(candidate['semantic_rank']):<11} | #{str(candidate['keyword_rank']):<11} | \033[1m{candidate['confidence']:.6f}\033[0m"
            print(line)

    print("\033[1;44m" + "="*80 + "\033[0m\n")

# ==============================================================================
# 7. BLOCCO DI ESECUZIONE PRINCIPALE
# ==============================================================================

if __name__ == "__main__":
    TOP_K_RESULTS = 5

    if not os.path.exists(HONEYPOT_CSV_FILE):
        print(f"\033[1;31m[ERRORE] File CSV '{HONEYPOT_CSV_FILE}' non trovato.\033[0m"); exit()

    analyzer, db_manager = initialize_system()

    print(f"\n\033[1;44m--- INIZIO ELABORAZIONE DATI --- \033[0m")
    try:
        df = pd.read_csv(HONEYPOT_CSV_FILE, delimiter=';', encoding='utf-8-sig', dtype={'called_number': str}, low_memory=False)
        print(f"\033[1;32m[DATI] Caricate {len(df)} righe totali dal CSV.\033[0m")

        if '@timestamp' in df.columns:
            if 'timestamp' in df.columns:
                df.drop(columns=['timestamp'], inplace=True)
            df.rename(columns={'@timestamp': 'timestamp'}, inplace=True)
            print("\033[1;33m[DATI] Colonna '@timestamp' impostata come 'timestamp' di riferimento.\033[0m")
        elif 'timestamp' not in df.columns:
            print(f"\033[1;31m[ERRORE] Nessuna colonna di timestamp ('@timestamp' o 'timestamp') trovata nel CSV.\033[0m"); exit()

        allowed_types = ['Cowrie', 'Honeytrap', 'Dionaea', 'Sentrypeer', 'Ciscoasa']
        df_filtered = df[df['type'].isin(allowed_types)].copy()
        print(f"\033[1;33m[DATI] Filtrate per tipi supportati -> {len(df_filtered)} righe.\033[0m")

        initial_rows = len(df_filtered)
        noise_patterns = ['Traceback', 'Exception occurred', 'Stopping server', 'Request timed out', 'ssl.SSLEOFError', 'socketserver.py', 'NameError', 'RecursionError']
        if 'message' in df_filtered.columns:
             for pattern in noise_patterns:
                df_filtered = df_filtered[~df_filtered['message'].str.contains(pattern, na=False)]
        print(f"\033[1;33m[DATI] Filtrate per rumore -> {len(df_filtered)} righe. ({initial_rows - len(df_filtered)} scartate)\033[0m")

        if 'timestamp' in df_filtered.columns:
            df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'], errors='coerce')

        for col in [c for c in ['dest_port', 'attack_connection.payload.length'] if c in df_filtered.columns]:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce').fillna(-1).astype(int)

        print(f"\033[1;32m[DATI] Tipi di dati convertiti.\033[0m")

    except Exception as e:
        print(f"\033[1;31m[ERRORE] Lettura/Processamento CSV fallito: {e}\033[0m"); traceback.print_exc(); exit()

    base_required_cols = ['type', 'timestamp', 'src_ip']
    initial_rows = len(df_filtered)
    df_filtered.dropna(subset=[c for c in base_required_cols if c in df_filtered.columns], inplace=True)
    print(f"\033[1;33m[DATI] Filtrate per valori essenziali mancanti -> {len(df_filtered)} righe. ({initial_rows - len(df_filtered)} scartate)\033[0m")

    if len(df_filtered) == 0: print(f"\033[1;31m[ERRORE] Nessuna riga valida trovata.\033[0m"); exit()
    print(f"\033[1;42m--- ELABORAZIONE DATI COMPLETATA --- \033[0m")

    sessions_to_process = defaultdict(list)

    if 'session' in df_filtered.columns:
        cowrie_df = df_filtered[(df_filtered['type'] == 'Cowrie') & df_filtered['session'].notna()].copy()
        if not cowrie_df.empty:
            print(f"\n\033[1;36m[RAGGRUPPAMENTO] Raggruppamento di {len(cowrie_df)} righe Cowrie per 'session'...\033[0m")
            for session_id, group in cowrie_df.groupby('session'):
                sessions_to_process[session_id] = group.sort_values(by='timestamp').to_dict('records')

    cisco_df = df_filtered[df_filtered['type'] == 'Ciscoasa'].copy()
    if not cisco_df.empty:
        print(f"\n\033[1;36m[RAGGRUPPAMENTO] Raggruppamento di {len(cisco_df)} righe CiscoASA per 'src_ip'...\033[0m")
        for src_ip, group in cisco_df.groupby('src_ip'):
            session_id = f"ciscoasa_{src_ip}"
            sessions_to_process[session_id] = group.sort_values(by='timestamp').to_dict('records')

    dionaea_df = df_filtered[df_filtered['type'] == 'Dionaea'].copy()
    if not dionaea_df.empty:
        print(f"\n\033[1;36m[RAGGRUPPAMENTO] Raggruppamento di {len(dionaea_df)} righe Dionaea per 'src_ip'...\033[0m")
        for src_ip, group in dionaea_df.groupby('src_ip'):
            session_id = f"dionaea_{src_ip}"
            sessions_to_process[session_id] = group.sort_values(by='timestamp').to_dict('records')

    honeytrap_df = df_filtered[df_filtered['type'] == 'Honeytrap'].copy()
    if not honeytrap_df.empty:
        print(f"\n\033[1;36m[RAGGRUPPAMENTO] Raggruppamento di {len(honeytrap_df)} righe Honeytrap per 'src_ip'...\033[0m")
        for src_ip, group in honeytrap_df.groupby('src_ip'):
            session_id = f"honeytrap_{src_ip}"
            sessions_to_process[session_id] = group.sort_values(by='timestamp').to_dict('records')

    sentrypeer_df = df_filtered[df_filtered['type'] == 'Sentrypeer'].copy()
    if not sentrypeer_df.empty:
        print(f"\n\033[1;36m[RAGGRUPPAMENTO] Raggruppamento di {len(sentrypeer_df)} righe SentryPeer per 'src_ip'...\033[0m")
        for src_ip, group in sentrypeer_df.groupby('src_ip'):
            session_id = f"sentrypeer_{src_ip}"
            sessions_to_process[session_id] = group.sort_values(by='timestamp').to_dict('records')

    print(f"\033[1;32m[RAGGRUPPAMENTO] Create {len(sessions_to_process)} sessioni totali da analizzare.\033[0m")

    analyzed_count = 0
    if sessions_to_process:
        for session_id, entries in tqdm(sessions_to_process.items(), desc="[ANALISI] Processo Sessioni Raggruppate"):
            h_type = entries[0]['type']
            print(f"\n\033[1;44m--- ANALISI SESSIONE: {session_id} ({h_type}) ---\033[0m")
            try:
                result = analyzer.analyze(generate_llm_input(entries, h_type), h_type, db_manager, top_k=TOP_K_RESULTS)
                result['src_ip'] = entries[0].get('src_ip', 'N/A')
                print_analysis_result(result, h_type, f"Sessione: {session_id}", TOP_K_RESULTS)
                analyzed_count += 1
            except Exception as e:
                print(f"\033[1;31m[ERRORE] Analisi sessione {session_id} fallita: {e}\033[0m"); traceback.print_exc()

    other_rows_df = df_filtered[~df_filtered['type'].isin(['Cowrie', 'Ciscoasa', 'Dionaea', 'Honeytrap', 'Sentrypeer'])]
    if not other_rows_df.empty:
        print(f"\n\033[1;36m[ANALISI] Processo di {len(other_rows_df)} eventi singoli...\033[0m")
        for index, entry in tqdm(other_rows_df.iterrows(), total=len(other_rows_df), desc="[ANALISI] Altri Tipi"):
            h_type = entry.get('type')
            print(f"\n\033[1;44m--- ANALISI RIGA: {index} ({h_type}) ---\033[0m")
            try:
                result = analyzer.analyze(generate_llm_input(entry.to_dict(), h_type), h_type, db_manager, top_k=TOP_K_RESULTS)
                result.update(entry.to_dict())
                print_analysis_result(result, h_type, f"Riga Index: {index}", TOP_K_RESULTS)
                analyzed_count += 1
            except Exception as e:
                print(f"\033[1;31m[ERRORE] Analisi riga {index} fallita: {e}\033[0m"); traceback.print_exc()

    print(f"\n\033[1;42m{'=' * 80}\033[0m")
    print(f"\033[1;32mANALISI GLOBALE COMPLETATA. {analyzed_count} eventi/sessioni analizzate.\033[0m")
    print(f"\033[1;42m{'=' * 80}\033[0m\n")
