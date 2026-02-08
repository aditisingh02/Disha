import { useState, useRef, useEffect } from 'react';
import { MessageCircle, Send, X, AlertTriangle, Info, AlertCircle, ThumbsUp, ThumbsDown } from 'lucide-react';
import { useChatStore } from '@/stores/chatStore';
import type { ChatMessage } from '@/stores/chatStore';
import { useChatSocket } from '@/hooks/useChatSocket';
import './ChatWidget.css';

export default function ChatWidget() {
    const [inputValue, setInputValue] = useState('');
    const [messageType, setMessageType] = useState<'info' | 'incident' | 'warning'>('info');
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const { messages, isConnected, username, isOpen, unreadCount, toggleOpen, setUsername } = useChatStore();
    const { sendMessage, sendReaction } = useChatSocket();

    const handleReaction = (messageId: string, reaction: 'up' | 'down') => {
        sendReaction({
            action: 'reaction',
            username,
            message_id: messageId,
            reaction,
        });
    };

    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSend = () => {
        if (!inputValue.trim()) return;

        sendMessage({
            username,
            message: inputValue.trim(),
            type: messageType,
        });

        setInputValue('');
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const formatTime = (timestamp: string) => {
        const date = new Date(timestamp);
        return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    };

    const getTypeIcon = (type: string) => {
        switch (type) {
            case 'incident': return <AlertTriangle size={14} className="text-red-500" />;
            case 'warning': return <AlertCircle size={14} className="text-yellow-500" />;
            default: return <Info size={14} className="text-blue-500" />;
        }
    };

    const getTypeClass = (type: string) => {
        switch (type) {
            case 'incident': return 'chat-message--incident';
            case 'warning': return 'chat-message--warning';
            default: return 'chat-message--info';
        }
    };

    return (
        <div className="chat-widget">
            {/* Floating Button */}
            <button
                onClick={toggleOpen}
                className="chat-fab"
                aria-label="Toggle chat"
            >
                {isOpen ? (
                    <X size={24} />
                ) : (
                    <>
                        <MessageCircle size={24} />
                        {unreadCount > 0 && (
                            <span className="chat-badge">{unreadCount > 99 ? '99+' : unreadCount}</span>
                        )}
                    </>
                )}
            </button>

            {/* Chat Panel */}
            {isOpen && (
                <div className="chat-panel">
                    {/* Header */}
                    <div className="chat-header">
                        <div className="chat-header__title">
                            <MessageCircle size={18} />
                            <span>Traffic Incidents</span>
                        </div>
                        <div className="chat-header__status">
                            <span className={`status-dot ${isConnected ? 'status-dot--online' : 'status-dot--offline'}`} />
                            <span>{isConnected ? 'Live' : 'Offline'}</span>
                        </div>
                    </div>

                    {/* Username Edit */}
                    <div className="chat-username">
                        <span>Your name:</span>
                        <input
                            type="text"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            maxLength={20}
                            className="chat-username__input"
                        />
                    </div>

                    {/* Messages */}
                    <div className="chat-messages">
                        {messages.length === 0 ? (
                            <div className="chat-empty">
                                <AlertCircle size={32} className="text-slate-400" />
                                <p>No messages yet</p>
                                <p className="text-sm">Report traffic incidents to help others!</p>
                            </div>
                        ) : (
                            messages.map((msg) => (
                                <div key={msg.id} className={`chat-message ${getTypeClass(msg.type)}`}>
                                    <div className="chat-message__header">
                                        {getTypeIcon(msg.type)}
                                        <span className="chat-message__username">{msg.username}</span>
                                        <span className="chat-message__time">{formatTime(msg.timestamp)}</span>
                                    </div>
                                    <p className="chat-message__text">{msg.message}</p>

                                    {/* Reaction Buttons */}
                                    <div className="chat-message__reactions">
                                        <button
                                            className="reaction-btn reaction-btn--up"
                                            onClick={() => handleReaction(msg.id, 'up')}
                                            title="Confirm incident"
                                        >
                                            <ThumbsUp size={14} />
                                            <span className="reaction-count">{msg.upvotes || 0}</span>
                                        </button>
                                        <button
                                            className="reaction-btn reaction-btn--down"
                                            onClick={() => handleReaction(msg.id, 'down')}
                                            title="Report as false"
                                        >
                                            <ThumbsDown size={14} />
                                            <span className="reaction-count">{msg.downvotes || 0}</span>
                                        </button>
                                    </div>
                                </div>
                            ))
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Input Area */}
                    <div className="chat-input-area">
                        {/* Message Type Selector */}
                        <div className="chat-type-selector">
                            <button
                                className={`chat-type-btn chat-type-btn--info ${messageType === 'info' ? 'chat-type-btn--active' : ''}`}
                                onClick={() => setMessageType('info')}
                                title="General Info"
                            >
                                <Info size={16} />
                            </button>
                            <button
                                className={`chat-type-btn chat-type-btn--warning ${messageType === 'warning' ? 'chat-type-btn--active' : ''}`}
                                onClick={() => setMessageType('warning')}
                                title="Warning"
                            >
                                <AlertCircle size={16} />
                            </button>
                            <button
                                className={`chat-type-btn chat-type-btn--incident ${messageType === 'incident' ? 'chat-type-btn--active' : ''}`}
                                onClick={() => setMessageType('incident')}
                                title="Incident/Accident"
                            >
                                <AlertTriangle size={16} />
                            </button>
                        </div>

                        {/* Input & Send */}
                        <div className="chat-input-row">
                            <input
                                type="text"
                                value={inputValue}
                                onChange={(e) => setInputValue(e.target.value)}
                                onKeyDown={handleKeyPress}
                                placeholder="Report traffic incident..."
                                className="chat-input"
                                disabled={!isConnected}
                            />
                            <button
                                onClick={handleSend}
                                disabled={!isConnected || !inputValue.trim()}
                                className="chat-send-btn"
                            >
                                <Send size={18} />
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
