
from __future__ import annotations

import asyncio
import base64
import dataclasses
import importlib
import inspect
import io
import json
import os
import queue
import random
import ssl
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple

# ----- Optional/External Libraries (loaded lazily) -----
try:
    import sounddevice as sd
    import numpy as np
except Exception:  # pragma: no cover
    sd = None
    np = None

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

try:
    import pyttsx3
except Exception:  # pragma: no cover
    pyttsx3 = None

try:
    import speech_recognition as sr
except Exception:  # pragma: no cover
    sr = None

try:
    from cryptography.fernet import Fernet
except Exception:  # pragma: no cover
    Fernet = None

# ==================== CORE DATA STRUCTURES ====================

class MessageType(Enum):
    VOICE_COMMAND = "voice_command"
    SYSTEM_EVENT = "system_event"
    RESPONSE_READY = "response_ready"
    ERROR = "error"

@dataclass
class Message:
    type: MessageType
    data: dict
    timestamp: float

@dataclass
class CommandResult:
    success: bool
    response: str
    actions: List[str]
    context: dict
    meta: Dict[str, Any] | None = None

@dataclass
class UserProfile:
    user_id: str
    voice_profile: Optional[bytes] = None  # serialized embedding
    preferences: Dict[str, Any] | None = None
    created_at: float | None = None

    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {
                "speech_rate": 180,
                "voice_id": "default",
                "output_mode": "SPEECH_PRIORITY",
                "wake_word": "hey ria",
            }
        if self.created_at is None:
            self.created_at = time.time()

# ==================== UTIL: PERSISTENCE & SECURITY ====================

class SecureStore:
    """Encrypted JSON store for user profiles and settings."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key_path = self.path.with_suffix(".key")
        self._fernet = None
        if Fernet:
            if self.key_path.exists():
                key = self.key_path.read_bytes()
            else:
                key = Fernet.generate_key()
                self.key_path.write_bytes(key)
            self._fernet = Fernet(key)

    def save(self, obj: Dict[str, Any]):
        raw = json.dumps(obj, ensure_ascii=False).encode()
        data = raw
        if self._fernet:
            data = self._fernet.encrypt(raw)
        self.path.write_bytes(data)

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        data = self.path.read_bytes()
        if self._fernet:
            try:
                data = self._fernet.decrypt(data)
            except Exception:
                pass
        try:
            return json.loads(data.decode())
        except Exception:
            return {}

# ==================== WAKE WORD DETECTOR ====================

class WakeWordDetector:
    """Energy-threshold wake detector. Replace with Porcupine/KWS later."""

    def __init__(self, wake_word: str = "hey ria", sensitivity: float = 0.5):
        self.wake_word = wake_word.lower()
        self.sensitivity = sensitivity
        self.is_listening = False
        self.callback: Optional[Callable[[], None]] = None
        self.energy_threshold = 300  # naive RMS-based

    def start_listening(self, callback: Callable[[], None]):
        self.callback = callback
        self.is_listening = True
        threading.Thread(target=self._audio_monitor, daemon=True).start()

    def _audio_monitor(self):
        if sd is None or np is None:
            print("[WakeWordDetector] sounddevice/numpy not available. Skipping audio monitor.")
            return
        try:
            with sd.InputStream(callback=self._audio_callback, channels=1, samplerate=16000):
                while self.is_listening:
                    time.sleep(0.1)
        except Exception as e:
            print(f"Audio monitoring error: {e}")

    def _audio_callback(self, indata, frames, time_info, status):
        if np is None:
            return
        if status:
            print(f"Audio status: {status}")
        energy = float(np.sqrt(np.mean(indata ** 2)) * 1000)
        if energy > self.energy_threshold and self.callback:
            print(f"Wake energy trigger: {energy:.2f}")
            self.callback()

# ==================== STT ENGINE ====================

class STTEngine:
    def __init__(self, offline_mode: bool = True):
        self.offline_mode = offline_mode
        self.recognizer = sr.Recognizer() if sr else None
        if self.recognizer:
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True

    def transcribe(self, audio_data=None) -> Optional[str]:
        if not self.recognizer:
            return None
        try:
            if audio_data is None:
                with sr.Microphone() as source:
                    print("Listening…")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            else:
                audio = audio_data
            if self.offline_mode:
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                except Exception:
                    text = self.recognizer.recognize_google(audio)
            else:
                text = self.recognizer.recognize_google(audio)
            return text.lower()
        except Exception as e:
            print(f"STT error: {e}")
            return None

# ==================== ADVANCED NLP ENGINE ====================

@dataclass
class IntentResult:
    intent: str
    confidence: float
    entities: Dict[str, Any]

class NLPEngine:
    """Hybrid intent classification: transformers -> spaCy -> regex/rules."""
    def __init__(self):
        try:
            self._TRANSFORMERS = importlib.import_module("transformers")
        except Exception:
            self._TRANSFORMERS = None
        try:
            self._SPACY = importlib.import_module("spacy")
        except Exception:
            self._SPACY = None
        self._nlp = None
        if self._SPACY:
            try:
                self._nlp = self._SPACY.load("en_core_web_sm")
            except Exception:
                self._nlp = None
        self._clf = None
        if self._TRANSFORMERS:
            try:
                self._clf = self._TRANSFORMERS.pipeline("zero-shot-classification")
            except Exception:
                self._clf = None
        # Add conversation intents
        self.intents = [
            "open_app", "close_app", "get_time", "get_date", "system_status",
            "cpu_usage", "memory_usage", "help", "exit", "weather", "search",
            "greeting", "farewell", "how_are_you", "who_are_you", "small_talk"
        ]

    def parse(self, text: str) -> Tuple[str, float, Dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return ("unknown", 0.0, {})
            
        # Convert to lowercase for easier matching
        l = text.lower()
        
        # 1) Transformers zero-shot (if available)
        if self._clf:
            try:
                out = self._clf(text, candidate_labels=self.intents, multi_label=False)
                label = out["labels"][0]
                score = float(out["scores"][0])
                return (label, score, self._extract_entities(text))
            except Exception:
                pass
                
        # 2) Conversation intents
        if any(word in l for word in ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]):
            return ("greeting", 0.8, {})
        if any(word in l for word in ["bye", "goodbye", "see you", "farewell", "quit", "exit"]):
            return ("farewell", 0.8, {})
        if any(word in l for word in ["how are you", "how do you feel", "what's up"]):
            return ("how_are_you", 0.8, {})
        if any(word in l for word in ["who are you", "what are you", "introduce yourself"]):
            return ("who_are_you", 0.8, {})
        if any(word in l for word in ["thank", "thanks", "appreciate"]):
            return ("small_talk", 0.7, {"subtype": "gratitude"})
        if any(word in l for word in ["sorry", "apologize"]):
            return ("small_talk", 0.7, {"subtype": "apology"})
            
        # 3) spaCy patterns
        if self._nlp:
            doc = self._nlp(text)
            lowered = text.lower()
            if "time" in lowered:
                return ("get_time", 0.6, {})
            if "date" in lowered or "today" in lowered:
                return ("get_date", 0.6, {})
            if "cpu" in lowered:
                return ("cpu_usage", 0.6, {})
            if "memory" in lowered or "ram" in lowered:
                return ("memory_usage", 0.6, {})
            if lowered.startswith("open "):
                return ("open_app", 0.6, {"app": lowered.split("open ",1)[1]})
            if lowered.startswith("close "):
                return ("close_app", 0.6, {"app": lowered.split("close ",1)[1]})
        # 4) Rules
        l = text.lower()
        if l.startswith("open "):
            return ("open_app", 0.51, {"app": l.split("open ",1)[1]})
        if l.startswith("close "):
            return ("close_app", 0.51, {"app": l.split("close ",1)[1]})
        if "weather" in l:
            return ("weather", 0.55, {})
        if "help" in l:
            return ("help", 0.55, {})
        if "exit" in l or "quit" in l:
            return ("exit", 0.55, {})
        if "system" in l:
            return ("system_status", 0.55, {})
        if "cpu" in l:
            return ("cpu_usage", 0.55, {})
        if "memory" in l:
            return ("memory_usage", 0.55, {})
        if "time" in l:
            return ("get_time", 0.55, {})
        if "date" in l or "today" in l:
            return ("get_date", 0.55, {})
        return ("unknown", 0.0, {})

    def _extract_entities(self, text: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        l = text.lower()
        for app in ["calculator", "notepad", "browser", "file explorer", "chrome", "edge"]:
            if app in l:
                out["app"] = app
                break
        return out

# ==================== VOICE BIOMETRICS ====================

class VoiceBiometrics:
    """Very lightweight speaker verification using MFCC embeddings (dev scaffold)."""

    def __init__(self):
        try:
            self._LIBROSA = importlib.import_module("librosa")
            self._np = importlib.import_module("numpy")
        except Exception:
            self._LIBROSA = None
            self._np = None
        self.enrolled: Dict[str, List[float]] = {}

    def enroll(self, user_id: str, wav_path: str) -> bool:
        if self._LIBROSA is None:
            print("[VoiceBiometrics] librosa not available; enrollment skipped.")
            return False
        try:
            y, sr = self._LIBROSA.load(wav_path, sr=16000)
            mfcc = self._LIBROSA.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            emb = mfcc.mean(axis=1).tolist()
            self.enrolled[user_id] = emb
            return True
        except Exception as e:
            print(f"[VoiceBiometrics] Enrollment error: {e}")
            return False

    def verify(self, user_id: str, wav_path: str, threshold: float = 15.0) -> bool:
        if self._LIBROSA is None or user_id not in self.enrolled:
            return True  # fallback: allow
        try:
            y, sr = self._LIBROSA.load(wav_path, sr=16000)
            mfcc = self._LIBROSA.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            emb = mfcc.mean(axis=1)
            ref = self._np.array(self.enrolled[user_id])  # type: ignore
            dist = float(self._np.linalg.norm(ref - emb))  # type: ignore
            print(f"[VoiceBiometrics] distance={dist:.2f}")
            return dist < threshold
        except Exception as e:
            print(f"[VoiceBiometrics] Verification error: {e}")
            return True

# ==================== MESSAGE BUS ====================

class MessageQueue:
    def __init__(self):
        self.queues = {msg_type: queue.Queue() for msg_type in MessageType}

    def publish(self, msg_type: MessageType, data: dict):
        message = Message(type=msg_type, data=data, timestamp=time.time())
        self.queues[msg_type].put(message)

    def subscribe(self, msg_type: MessageType, timeout=1.0) -> Optional[Message]:
        try:
            return self.queues[msg_type].get(timeout=timeout)
        except queue.Empty:
            return None

# ==================== PLUGIN SYSTEM ====================

class Plugin(Protocol):
    name: str
    description: str
    permissions: List[str]

    def can_handle(self, intent: str, entities: Dict[str, Any]) -> bool: ...
    def handle(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> CommandResult: ...

class PluginManager:
    def __init__(self):
        self._plugins: List[Plugin] = []

    def register(self, plugin: Plugin):
        self._plugins.append(plugin)

    def discover(self, module_names: Iterable[str]):
        for mod in module_names:
            try:
                m = importlib.import_module(mod)
                for _, obj in inspect.getmembers(m, inspect.isclass):
                    if hasattr(obj, "handle") and hasattr(obj, "can_handle"):
                        self.register(obj())
            except Exception as e:
                print(f"[PluginManager] Failed to import {mod}: {e}")

    def route(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> Optional[CommandResult]:
        for p in self._plugins:
            try:
                if p.can_handle(intent, entities):
                    return p.handle(intent, entities, context)
            except Exception as e:
                print(f"[Plugin:{getattr(p,'name','?')}] error: {e}")
        return None

# ----- Built-in plugins -----

class CalculatorPlugin:
    name = "calculator"
    description = "Performs arithmetic and scientific math"
    permissions = []

    def can_handle(self, intent: str, entities: Dict[str, Any]) -> bool:
        text = entities.get("text", "")
        if not text:
            return False
        l = text.lower()
        return intent == "unknown" and any(op in l for op in ["+","-","*","/","(",")","sqrt","sin","cos","tan","log","abs","round"]) 

    def handle(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        import math
        expr = entities.get("text", "")
        try:
            # Sanitize expression
            expr = "".join(c for c in expr if c in "0123456789+-*/()., eEabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
            allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            allowed["abs"] = abs
            allowed["round"] = round
            value = eval(expr, {"__builtins__": {}}, allowed)
            return CommandResult(True, f"Result: {value}", ["calc"], context)
        except Exception as e:
            return CommandResult(False, f"I couldn't compute that expression: {e}", [], context)

class WeatherPlugin:
    name = "weather"
    description = "Returns weather for a city (stubbed)"
    permissions = ["network"]

    def can_handle(self, intent: str, entities: Dict[str, Any]) -> bool:
        return intent == "weather"

    def handle(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        city = entities.get("city", context.get("default_city", "Kampala"))
        return CommandResult(True, f"Weather in {city}: 24°C, partly cloudy (stub)", ["weather"], context)

class JokesPlugin:
    name = "jokes"
    description = "Tells a random programming joke"
    permissions = []

    def can_handle(self, intent: str, entities: Dict[str, Any]) -> bool:
        text = entities.get("text", "")
        if not text:
            return False
        return "joke" in text.lower()

    def handle(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        jokes = [
            "Why don’t programmers like nature? Too many bugs.",
            "I told my computer I needed a break—now it won’t stop sending me Kit-Kats.",
            "Why do Java developers wear glasses? Because they don’t C#.",
        ]
        return CommandResult(True, random.choice(jokes), ["joke"], context)

class WebSearchPlugin:
    name = "web_search"
    description = "Performs web searches (Google CSE if keys provided; fallback to DuckDuckGo Instant Answer)"
    permissions = ["internet"]

    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.cx_id = os.environ.get("GOOGLE_CSE_ID")
        self.has_google = bool(self.api_key and self.cx_id)

    def can_handle(self, intent: str, entities: Dict[str, Any]) -> bool:
        text = entities.get("text", "")
        if not text:
            return False
        t = text.lower()
        return (intent == "search" or 
                t.startswith("search ") or 
                t.startswith("google ") or 
                t.startswith("look up ") or
                "news" in t or
                "information about" in t or
                "tell me about" in t)

    def _google_search(self, query: str) -> List[Dict[str, Any]]:
        try:
            if not self.has_google:
                return []
            import requests
            url = "https://www.googleapis.com/customsearch/v1"
            res = requests.get(url, params={
                "q": query, 
                "cx": self.cx_id, 
                "key": self.api_key, 
                "num": 5
            }, timeout=10)
            data = res.json()
            return data.get("items", [])
        except Exception as e:
            print(f"[WebSearchPlugin] Google error: {e}")
            return []

    def _ddg(self, query: str) -> Dict[str, Any]:
        try:
            import requests
            res = requests.get("https://api.duckduckgo.com/", params={
                "q": query, 
                "format": "json", 
                "no_html": 1, 
                "no_redirect": 1
            }, timeout=10)
            return res.json()
        except Exception as e:
            print(f"[WebSearchPlugin] DDG error: {e}")
            return {}

    def handle(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        text = entities.get("text", "")
        q = text
        
        # Extract the search query from various patterns
        patterns = ["search ", "google ", "news ", "look up ", "information about ", "tell me about "]
        for p in patterns:
            if q.lower().startswith(p):
                q = q[len(p):].strip()
                break
                
        # Use Google if available (better results when connected to internet)
        if self.has_google:
            results = self._google_search(q)
            if results:
                # Format a more conversational response
                top_result = results[0]
                title = top_result.get('title', '')
                snippet = top_result.get('snippet', '')
                link = top_result.get('link', '')
                
                msg = f"According to my search: {snippet}"
                if title:
                    msg = f"Here's what I found about {title}: {snippet}"
                
                return CommandResult(
                    True, 
                    msg, 
                    ["web_search"], 
                    context, 
                    {
                        "provider": "google", 
                        "query": q, 
                        "items": results,
                        "top_result": top_result
                    }
                )
        
        # Fallback to DuckDuckGo
        ddg = self._ddg(q)
        abstract = ddg.get("AbstractText")
        if abstract and abstract != "":
            response = f"Here's what I found: {abstract}"
        else:
            response = "I found some results but couldn't get a concise answer. You might want to try a different search query."
            
        return CommandResult(
            True, 
            response, 
            ["web_search"], 
            context, 
            {
                "provider": "ddg", 
                "query": q, 
                "data": ddg
            }
        )

# Add a new ConversationPlugin
class ConversationPlugin:
    name = "conversation"
    description = "Handles greetings, farewells, and small talk"
    permissions = []

    def can_handle(self, intent: str, entities: Dict[str, Any]) -> bool:
        return intent in ["greeting", "farewell", "how_are_you", "who_are_you", "small_talk"]

    def handle(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        if intent == "greeting":
            greetings = [
                "Hello there! How can I help you today?",
                "Hi! What can I do for you?",
                "Hey! Nice to hear from you. What would you like to do?",
                "Greetings! I'm here and ready to assist you."
            ]
            response = random.choice(greetings)
            
        elif intent == "farewell":
            farewells = [
                "Goodbye! Have a great day!",
                "See you later! Don't hesitate to call if you need anything.",
                "Farewell! It was nice talking with you.",
                "Bye for now! I'll be here when you need me."
            ]
            response = random.choice(farewells)
            
        elif intent == "how_are_you":
            feelings = [
                "I'm functioning well, thank you for asking! How can I help you today?",
                "I'm doing great! Ready to assist with whatever you need.",
                "All systems are operational! What can I do for you?",
                "I'm good, thanks! How about you?"
            ]
            response = random.choice(feelings)
            
        elif intent == "who_are_you":
            introductions = [
                "I'm Ria, your AI assistant. I'm here to help with tasks, information, and conversations!",
                "I'm Ria, a digital assistant designed to make your life easier. How can I help?",
                "You can call me Ria! I'm an AI assistant ready to help with various tasks and questions."
            ]
            response = random.choice(introductions)
            
        elif intent == "small_talk":
            subtype = entities.get("subtype", "")
            if subtype == "gratitude":
                response = "You're welcome! Is there anything else I can help with?"
            elif subtype == "apology":
                response = "No need to apologize! How can I help you?"
            else:
                responses = [
                    "That's interesting! What else would you like to talk about?",
                    "I see. Is there something specific you'd like help with?",
                    "Interesting! Did you have a question or task for me?"
                ]
                response = random.choice(responses)
                
        return CommandResult(True, response, ["conversation"], context)

# ==================== COMMAND PROCESSOR ====================

class CommandProcessor:
    def __init__(self, nlp: NLPEngine, plugins: PluginManager):
        self.nlp = nlp
        self.plugins = plugins
        self.context: Dict[str, Any] = {"conversation": []}
        self.commands = self._initialize_commands()

    def _initialize_commands(self) -> Dict[str, Callable[[str, Dict[str, Any]], CommandResult]]:
        commands = {
            "open_app": self._open_application,
            "close_app": self._close_application,
            "get_time": self._get_time,
            "get_date": self._get_date,
            "system_status": self._system_status,
            "cpu_usage": self._cpu_usage,
            "memory_usage": self._memory_usage,
            "help": self._show_help,
            "exit": self._exit_app,
        }
        return commands

    def process(self, text: str) -> CommandResult:
        intent, conf, entities = self.nlp.parse(text)
        entities.setdefault("text", text)
        self.context["conversation"].append({"text": text, "intent": intent, "conf": conf})
        if intent in self.commands:
            return self.commands[intent](text, entities)
        plugin_res = self.plugins.route(intent, entities, self.context)
        if plugin_res:
            return plugin_res
        return CommandResult(False, f"I didn't understand: '{text}'. Say 'help' to see options.", [], self.context, {"intent": intent, "conf": conf})

    # ----- Built-ins -----
    def _open_application(self, text: str, entities: Dict[str, Any]) -> CommandResult:
        app_name = entities.get("app") or text.replace("open", "").strip()
        apps = {
            "calculator": "calc.exe",
            "notepad": "notepad.exe",
            "browser": "chrome.exe",
            "chrome": "chrome.exe",
            "edge": "msedge.exe",
            "file explorer": "explorer.exe",
            "netflix": "netflix.exe",
            "apple tv": "AppleTV.exe",
            "apple music": "AppleMusic.exe",
            "microsoft store": "ms-windows-store://home",
            "microsoft office": "ms-office://",
        }
        exe = apps.get(app_name)
        if exe:
            try:
                import subprocess
                subprocess.Popen(exe)
                return CommandResult(True, f"Opening {app_name}", [f"open_{app_name}"], self.context)
            except Exception as e:
                return CommandResult(False, f"Failed to open {app_name}: {e}", [], self.context)
        return CommandResult(False, f"App '{app_name}' not found in list.", [], self.context)

    def _close_application(self, text: str, entities: Dict[str, Any]) -> CommandResult:
        return CommandResult(True, "Close command received (implement process kill by name).", ["close_application"], self.context)

    def _get_time(self, text: str, entities: Dict[str, Any]) -> CommandResult:
        current_time = time.strftime("%I:%M %p")
        return CommandResult(True, f"The current time is {current_time}", ["get_time"], self.context)

    def _get_date(self, text: str, entities: Dict[str, Any]) -> CommandResult:
        current_date = time.strftime("%A, %B %d, %Y")
        return CommandResult(True, f"Today is {current_date}", ["get_date"], self.context)

    def _system_status(self, text: str, entities: Dict[str, Any]) -> CommandResult:
        if psutil is None:
            return CommandResult(True, "System status unavailable (psutil missing).", ["system_status"], self.context)
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        return CommandResult(True, f"System status: CPU {cpu}%, Memory {mem}%", ["system_status"], self.context)

    def _cpu_usage(self, text: str, entities: Dict[str, Any]) -> CommandResult:
        if psutil is None:
            return CommandResult(True, "CPU usage unavailable (psutil missing).", ["cpu_usage"], self.context)
        cpu = psutil.cpu_percent()
        return CommandResult(True, f"CPU usage is {cpu}%", ["cpu_usage"], self.context)

    def _memory_usage(self, text: str, entities: Dict[str, Any]) -> CommandResult:
        if psutil is None:
            return CommandResult(True, "Memory usage unavailable (psutil missing).", ["memory_usage"], self.context)
        mem = psutil.virtual_memory().percent
        return CommandResult(True, f"Memory usage is {mem}%", ["memory_usage"], self.context)

    def _show_help(self, text: str, entities: Dict[str, Any]) -> CommandResult:
        help_text = (
            "Available: open <app>, close <app>, time, date, system, cpu, memory, exit. "
            "Plugins: calculator (say '3+4'), weather (say 'weather Kampala'), "
            "web search (say 'search AI news' or 'news Uganda'), jokes ('tell me a joke'). "
            "Conversation: hello, how are you, who are you, goodbye."
        )
        return CommandResult(True, help_text, ["show_help"], self.context)

    def _exit_app(self, text: str, entities: Dict[str, Any]) -> CommandResult:
        return CommandResult(True, "Goodbye!", ["exit_application"], self.context)

# ==================== RESPONSE SYSTEM ====================

class ResponseSystem:
    def __init__(self):
        self.tts_engine = pyttsx3.init() if pyttsx3 else None
        if self.tts_engine:
            try:
                voices = self.tts_engine.getProperty('voices')
                if voices:
                    self.tts_engine.setProperty('voice', voices[0].id)
            except Exception as e:
                print(f"TTS voice setup error: {e}")
        self.output_modes = {
            "SPEECH_PRIORITY": {"speech": True, "text": True},
            "QUIET_MODE": {"speech": False, "text": True},
            "TEXT_ONLY": {"speech": False, "text": True},
            "SPEECH_ONLY": {"speech": True, "text": False},
        }
        self.current_mode = "SPEECH_PRIORITY"

    def generate(self, message: str, context: dict | None = None):
        cfg = self.output_modes.get(self.current_mode, self.output_modes["SPEECH_PRIORITY"])
        if cfg["text"]:
            print(f"Ria: {message}")
        if cfg["speech"] and self.tts_engine:
            try:
                self.tts_engine.say(message)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")

# ==================== CLOUD INTEGRATION (stubs) ====================

class CloudClient:
    def __init__(self, base_url: str | None = None, token: str | None = None):
        self.base_url = base_url or "https://api.example.com"
        self.token = token or "dev-token"
        self.session = None

    def http_get(self, path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        print(f"[CloudClient] GET {self.base_url}{path} params={params}")
        return {"ok": True, "data": {}}

    def http_post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[CloudClient] POST {self.base_url}{path} payload_keys={list(payload.keys())}")
        return {"ok": True}

    async def ws_listen(self, path: str, on_message: Callable[[Dict[str, Any]], None]):
        for i in range(3):
            await asyncio.sleep(2)
            on_message({"type": "ping", "i": i})

# ==================== SECURITY MANAGER ====================

class SecurityManager:
    def __init__(self, store: SecureStore):
        self.store = store
        self.authenticated = False
        self._profiles: Dict[str, Any] = self.store.load()

    def encrypt_data(self, data: str) -> bytes:
        return data.encode()

    def authenticate_user(self, user_id: str = "default") -> bool:
        self.authenticated = True
        prof = self._profiles.get(user_id) or {}
        self._profiles.setdefault(user_id, prof)
        self.store.save(self._profiles)
        return True

    def save_profile(self, user_id: str, profile: UserProfile):
        self._profiles[user_id] = dataclasses.asdict(profile)
        self.store.save(self._profiles)

    def load_profile(self, user_id: str) -> UserProfile:
        data = self._profiles.get(user_id) or {}
        voice_profile = None
        if data.get("voice_profile"):
            try:
                voice_profile = base64.b64decode(data.get("voice_profile", ""))
            except Exception:
                pass
        return UserProfile(user_id=user_id,
                           voice_profile=voice_profile,
                           preferences=data.get("preferences"),
                           created_at=data.get("created_at"))

# ==================== THOUGHT ENGINE ====================

class ThoughtEngine:
    def reflect(self, user_text: str, result: CommandResult, ctx: Dict[str, Any]) -> Optional[str]:
        t = (user_text or "").lower()
        if result.success:
            if any(k in t for k in ["news", "search"]):
                return "Want me to keep watching this topic and ping you with big updates?"
            if any(op in t for op in ["+","-","*","/"]):
                return "I can stay in calculator mode—just keep typing expressions."
            if "weather" in t:
                city = ctx.get("default_city", "your city")
                return f"Should I set {city} as your default city for weather?"
            if any(k in t for k in ["time","date"]):
                return "I can also set reminders—say 'remind me'."
            # Add conversation reflections
            if any(word in t for word in ["hello", "hi", "hey", "greetings"]):
                return "It's nice when users greet me politely. Makes the"


# ==================== THOUGHT ENGINE ====================

class ThoughtEngine:
    def reflect(self, user_text: str, result: CommandResult, ctx: Dict[str, Any]) -> Optional[str]:
        t = (user_text or "").lower()
        if result.success:
            if any(k in t for k in ["news", "search"]):
                return "Want me to keep watching this topic and ping you with big updates?"
            if any(op in t for op in ["+","-","*","/"]):
                return "I can stay in calculator mode—just keep typing expressions."
            if "weather" in t:
                city = ctx.get("default_city", "your city")
                return f"Should I set {city} as your default city for weather?"
            if any(k in t for k in ["time","date"]):
                return "I can also set reminders—say 'remind me'."
        else:
            return "Not sure I got that. Say 'help' to see examples."
        return None

# ==================== MAIN RIA SYSTEM ====================

class RiaSystem:
    def __init__(self, offline_mode: bool = True, user_id: str = "default"):
        self.offline_mode = offline_mode
        self.is_running = False
        self.user_id = user_id

        # Storage & Security
        data_dir = Path.home() / ".ria"
        self.secure_store = SecureStore(data_dir / "profiles.json")
        self.security_manager = SecurityManager(self.secure_store)

        # Components
        self.message_queue = MessageQueue()
        self.wake_detector = WakeWordDetector()
        self.stt_engine = STTEngine(offline_mode)
        self.nlp = NLPEngine()
        self.plugins = PluginManager()
        self.command_processor = CommandProcessor(self.nlp, self.plugins)
        self.response_system = ResponseSystem()
        self.voice_bio = VoiceBiometrics()
        self.cloud = CloudClient()
        self.thoughts = ThoughtEngine()

        # User management
        self.current_user: Optional[UserProfile] = None

        # Register plugins
        self.plugins.register(CalculatorPlugin())
        self.plugins.register(WeatherPlugin())
        self.plugins.register(JokesPlugin())
        self.plugins.register(WebSearchPlugin())

    # ---------- Lifecycle ----------
    def start(self):
        print("Starting Ria AI Assistant (v2)…")
        self.is_running = True
        if self.security_manager.authenticate_user(self.user_id):
            print("User authenticated successfully")
            self.current_user = self.security_manager.load_profile(self.user_id)
        else:
            print("Authentication failed")
            return
        self.wake_detector.start_listening(self._on_wake_word_detected)
        self._main_loop()

    def stop(self):
        print("Stopping Ria AI Assistant…")
        self.is_running = False
        self.wake_detector.is_listening = False

    # ---------- Wake/Command Flow ----------
    def _on_wake_word_detected(self):
        print("Wake word detected! Listening for command…")
        self.message_queue.publish(MessageType.VOICE_COMMAND, {"action": "start_listening"})

    def _process_voice_command(self):
        try:
            text = self.stt_engine.transcribe()
            if text:
                print(f"User said: {text}")
                result = self.command_processor.process(text)
                self.response_system.generate(result.response)
                thought = self.thoughts.reflect(text, result, self.command_processor.context)
                if thought:
                    self.response_system.generate("(thinking) " + thought)
                if "exit_application" in result.actions:
                    self.stop()
            else:
                self.response_system.generate("I didn't catch that. Could you repeat?")
        except Exception as e:
            print(f"Error processing voice command: {e}")
            self.response_system.generate("I encountered an error. Please try again.")

    def _main_loop(self):
        print("Ria is ready! Say the wake word or type in interactive mode.")
        while self.is_running:
            try:
                message = self.message_queue.subscribe(MessageType.VOICE_COMMAND, timeout=0.1)
                if message and message.type == MessageType.VOICE_COMMAND:
                    self._process_voice_command()
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nShutting down…")
                self.stop()
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(1)

    # ---------- Enrollment/Verification ----------
    def enroll_voice(self, wav_path: str):
        if self.voice_bio.enroll(self.user_id, wav_path):
            emb = self.voice_bio.enrolled.get(self.user_id)
            if emb is not None:
                prof = self.security_manager.load_profile(self.user_id)
                # store raw floats as JSON for simplicity
                prof.voice_profile = json.dumps(emb).encode()
                self.security_manager.save_profile(self.user_id, prof)
            print("Enrollment done.")
        else:
            print("Enrollment skipped (librosa unavailable).")

    def verify_voice(self, wav_path: str) -> bool:
        ok = self.voice_bio.verify(self.user_id, wav_path)
        print(f"Voice verify -> {ok}")
        return ok

# ==================== UTILITY MODES ====================

def test_voice_commands():
    ria = RiaSystem(offline_mode=True)
    ria.security_manager.authenticate_user()
    tests = [
        "time", "date", "system", "cpu", "memory", "help", 
        "open calculator", "3+4*2", "weather Kampala",
        "tell me a joke", "search AI news", "news Uganda technology",
    ]
    for t in tests:
        print(f"\n> {t}")
        res = ria.command_processor.process(t)
        ria.response_system.generate(res.response)
        thought = ria.thoughts.reflect(t, res, ria.command_processor.context)
        if thought:
            ria.response_system.generate("(thinking) " + thought)
        time.sleep(0.2)


def interactive_mode():
    ria = RiaSystem(offline_mode=True)
    ria.security_manager.authenticate_user()
    print("Interactive mode. Type commands, 'exit' to quit.")
    while True:
        try:
            command = input("\nYou: ").strip()
            if command.lower() in ["exit", "quit", "bye"]:
                break
            res = ria.command_processor.process(command)
            ria.response_system.generate(res.response)
            thought = ria.thoughts.reflect(command, res, ria.command_processor.context)
            if thought:
                ria.response_system.generate("(thinking) " + thought)
            if "exit_application" in res.actions:
                break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

# ==================== MAIN ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ria AI Assistant – Advanced Prototype (v2)")
    parser.add_argument("--mode", choices=["full", "test", "interactive"], default="interactive")
    parser.add_argument("--offline", action="store_true", default=True, help="Use offline STT mode")
    args = parser.parse_args()

    if args.mode == "full":
        ria = RiaSystem(offline_mode=args.offline)
        ria.start()
    elif args.mode == "test":
        test_voice_commands()
    else:
        interactive_mode()

