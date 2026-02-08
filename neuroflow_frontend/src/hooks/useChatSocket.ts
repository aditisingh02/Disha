import { useEffect, useRef, useCallback } from 'react';
import { useChatStore, ChatMessage } from '@/stores/chatStore';
import { CHAT_WS_URL } from '@/utils/constants';

const RECONNECT_BASE_MS = 2000;
const RECONNECT_MAX_MS = 30_000;

interface ChatPayload {
    action?: string;
    username: string;
    message: string;
    type: 'info' | 'incident' | 'warning';
}

interface ReactionPayload {
    action: 'reaction';
    username: string;
    message_id: string;
    reaction: 'up' | 'down';
}

export function useChatSocket() {
    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
    const backoffRef = useRef(RECONNECT_BASE_MS);

    const addMessage = useChatStore((s) => s.addMessage);
    const setMessages = useChatStore((s) => s.setMessages);
    const updateReaction = useChatStore((s) => s.updateReaction);
    const setConnected = useChatStore((s) => s.setConnected);

    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) return;

        try {
            const ws = new WebSocket(CHAT_WS_URL);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('[ChatWS] Connected to chat server');
                setConnected(true);
                backoffRef.current = RECONNECT_BASE_MS;
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);

                    if (data.event === 'message') {
                        // Single new message
                        addMessage(data.data as ChatMessage);
                    } else if (data.event === 'history') {
                        // Message history on connect
                        setMessages(data.messages as ChatMessage[]);
                    } else if (data.event === 'reaction') {
                        // Reaction update
                        updateReaction(
                            data.data.message_id,
                            data.data.upvotes,
                            data.data.downvotes
                        );
                    }
                } catch {
                    console.error('[ChatWS] Failed to parse message');
                }
            };

            ws.onclose = () => {
                setConnected(false);
                const delay = backoffRef.current;
                backoffRef.current = Math.min(delay * 2, RECONNECT_MAX_MS);
                console.log(`[ChatWS] Disconnected — reconnecting in ${(delay / 1000).toFixed(0)}s...`);
                reconnectTimer.current = setTimeout(connect, delay);
            };

            ws.onerror = () => {
                ws.close();
            };
        } catch {
            const delay = backoffRef.current;
            backoffRef.current = Math.min(delay * 2, RECONNECT_MAX_MS);
            reconnectTimer.current = setTimeout(connect, delay);
        }
    }, [addMessage, setMessages, updateReaction, setConnected]);

    const sendMessage = useCallback((payload: ChatPayload) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ ...payload, action: 'message' }));
        } else {
            console.error('[ChatWS] Cannot send - not connected');
        }
    }, []);

    const sendReaction = useCallback((payload: ReactionPayload) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(payload));
        } else {
            console.error('[ChatWS] Cannot react - not connected');
        }
    }, []);

    useEffect(() => {
        connect();
        return () => {
            clearTimeout(reconnectTimer.current);
            wsRef.current?.close();
        };
    }, [connect]);

    return { sendMessage, sendReaction };
}
