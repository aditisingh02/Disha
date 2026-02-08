"""
NeuroFlow BharatFlow — Real-Time Incident Chat
WebSocket endpoint for live traffic incident reporting.
Users can report accidents, congestion, and road hazards in real-time.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import List, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger("neuroflow.chat")

router = APIRouter(prefix="/api/v1", tags=["Chat"])


# ═══════════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    """A single chat message with vote counts."""
    id: str
    username: str
    message: str
    timestamp: datetime
    type: str = "info"  # info, incident, warning
    upvotes: int = 0
    downvotes: int = 0


class ChatPayload(BaseModel):
    """Incoming message from client."""
    action: str = "message"  # "message" or "reaction"
    username: str
    message: str = ""
    type: str = "info"
    # For reactions
    message_id: str = ""
    reaction: str = ""  # "up" or "down"


# ═══════════════════════════════════════════════════════════════
# Chat Manager (Singleton)
# ═══════════════════════════════════════════════════════════════

class ChatManager:
    """
    Manages WebSocket connections and broadcasts messages to all clients.
    Tracks message reactions (upvotes/downvotes) per user.
    """
    
    def __init__(self):
        self._clients: Set[WebSocket] = set()
        self._messages: List[ChatMessage] = []  # Recent message history
        self._message_map: dict[str, ChatMessage] = {}  # Quick lookup by ID
        self._reactions: dict[str, dict[str, str]] = {}  # {msg_id: {username: "up"/"down"}}
        self._max_history = 100  # Keep last 100 messages
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection and send recent history."""
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)
        
        # Send recent message history to new client
        if self._messages:
            await websocket.send_json({
                "event": "history",
                "messages": [m.model_dump(mode="json") for m in self._messages[-50:]]
            })
        
        logger.info(f"Chat client connected. Total clients: {len(self._clients)}")
    
    async def disconnect(self, websocket: WebSocket):
        """Remove disconnected client."""
        async with self._lock:
            self._clients.discard(websocket)
        logger.info(f"Chat client disconnected. Total clients: {len(self._clients)}")
    
    async def broadcast(self, message: ChatMessage):
        """Send message to all connected clients."""
        # Store in history
        self._messages.append(message)
        self._message_map[message.id] = message
        self._reactions[message.id] = {}  # Initialize reactions for this message
        
        if len(self._messages) > self._max_history:
            # Remove oldest messages from map and reactions
            for old_msg in self._messages[:-self._max_history]:
                self._message_map.pop(old_msg.id, None)
                self._reactions.pop(old_msg.id, None)
            self._messages = self._messages[-self._max_history:]
        
        # Broadcast to all clients
        payload = {
            "event": "message",
            "data": message.model_dump(mode="json")
        }
        
        await self._send_to_all(payload)
    
    async def handle_reaction(self, message_id: str, username: str, reaction: str) -> bool:
        """Handle upvote/downvote reaction. Returns True if successful."""
        if message_id not in self._message_map:
            return False
        
        message = self._message_map[message_id]
        user_reactions = self._reactions.get(message_id, {})
        previous_reaction = user_reactions.get(username)
        
        # If user already voted the same way, remove their vote (toggle off)
        if previous_reaction == reaction:
            if reaction == "up":
                message.upvotes = max(0, message.upvotes - 1)
            else:
                message.downvotes = max(0, message.downvotes - 1)
            del user_reactions[username]
        else:
            # Remove previous opposite vote if exists
            if previous_reaction == "up":
                message.upvotes = max(0, message.upvotes - 1)
            elif previous_reaction == "down":
                message.downvotes = max(0, message.downvotes - 1)
            
            # Add new vote
            if reaction == "up":
                message.upvotes += 1
            else:
                message.downvotes += 1
            user_reactions[username] = reaction
        
        self._reactions[message_id] = user_reactions
        
        # Broadcast updated reaction counts
        await self._send_to_all({
            "event": "reaction",
            "data": {
                "message_id": message_id,
                "upvotes": message.upvotes,
                "downvotes": message.downvotes,
            }
        })
        
        return True
    
    async def _send_to_all(self, payload: dict):
        """Send payload to all connected clients."""
        disconnected = []
        for client in self._clients.copy():
            try:
                await client.send_json(payload)
            except Exception:
                disconnected.append(client)
        
        for client in disconnected:
            await self.disconnect(client)
    
    @property
    def client_count(self) -> int:
        return len(self._clients)


# Global chat manager instance
chat_manager = ChatManager()


# ═══════════════════════════════════════════════════════════════
# WebSocket Endpoint
# ═══════════════════════════════════════════════════════════════

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time incident chat.
    
    Client sends:
    {
        "username": "User123",
        "message": "Accident on PIE near Eunos",
        "type": "incident"  // "info" | "incident" | "warning"
    }
    
    Server broadcasts to all clients:
    {
        "event": "message",
        "data": {
            "id": "uuid",
            "username": "User123",
            "message": "Accident on PIE near Eunos",
            "timestamp": "2024-01-01T12:00:00",
            "type": "incident"
        }
    }
    """
    await chat_manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            try:
                payload = ChatPayload(**data)
                
                if payload.action == "reaction":
                    # Handle upvote/downvote reaction
                    if payload.message_id and payload.reaction in ("up", "down"):
                        success = await chat_manager.handle_reaction(
                            payload.message_id,
                            payload.username,
                            payload.reaction
                        )
                        if not success:
                            await websocket.send_json({
                                "event": "error",
                                "message": "Message not found for reaction"
                            })
                else:
                    # Handle new message
                    if payload.message.strip():
                        message = ChatMessage(
                            id=str(uuid.uuid4()),
                            username=payload.username,
                            message=payload.message,
                            timestamp=datetime.utcnow(),
                            type=payload.type,
                        )
                        await chat_manager.broadcast(message)
                        logger.info(f"Chat: [{message.type}] {message.username}: {message.message[:50]}")
                
            except Exception as e:
                # Send error back to sender
                await websocket.send_json({
                    "event": "error",
                    "message": f"Invalid message format: {str(e)}"
                })
    
    except WebSocketDisconnect:
        await chat_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Chat WebSocket error: {e}")
        await chat_manager.disconnect(websocket)


# ═══════════════════════════════════════════════════════════════
# REST Endpoints (for debugging/admin)
# ═══════════════════════════════════════════════════════════════

@router.get("/chat/status")
async def get_chat_status():
    """Get current chat status."""
    return {
        "connected_clients": chat_manager.client_count,
        "message_history_count": len(chat_manager._messages),
    }
