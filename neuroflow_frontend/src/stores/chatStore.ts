import { create } from 'zustand';

export interface ChatMessage {
    id: string;
    username: string;
    message: string;
    timestamp: string;
    type: 'info' | 'incident' | 'warning';
    upvotes: number;
    downvotes: number;
}

interface ChatState {
    messages: ChatMessage[];
    isConnected: boolean;
    username: string;
    isOpen: boolean;
    unreadCount: number;

    addMessage: (msg: ChatMessage) => void;
    setMessages: (msgs: ChatMessage[]) => void;
    updateReaction: (messageId: string, upvotes: number, downvotes: number) => void;
    setConnected: (connected: boolean) => void;
    setUsername: (name: string) => void;
    toggleOpen: () => void;
    setOpen: (open: boolean) => void;
    resetUnread: () => void;
}

// Generate a random username for new users
const generateUsername = (): string => {
    const adjectives = ['Swift', 'Smart', 'Quick', 'Alert', 'Sharp'];
    const nouns = ['Driver', 'Rider', 'Commuter', 'Traveler', 'User'];
    const adj = adjectives[Math.floor(Math.random() * adjectives.length)];
    const noun = nouns[Math.floor(Math.random() * nouns.length)];
    const num = Math.floor(Math.random() * 1000);
    return `${adj}${noun}${num}`;
};

export const useChatStore = create<ChatState>((set, get) => ({
    messages: [],
    isConnected: false,
    username: generateUsername(),
    isOpen: false,
    unreadCount: 0,

    addMessage: (msg) => set((state) => ({
        messages: [...state.messages.slice(-99), {
            ...msg,
            upvotes: msg.upvotes ?? 0,
            downvotes: msg.downvotes ?? 0
        }],
        unreadCount: state.isOpen ? 0 : state.unreadCount + 1,
    })),

    setMessages: (msgs) => set({
        messages: msgs.map(m => ({
            ...m,
            upvotes: m.upvotes ?? 0,
            downvotes: m.downvotes ?? 0
        }))
    }),

    updateReaction: (messageId, upvotes, downvotes) => set((state) => ({
        messages: state.messages.map(msg =>
            msg.id === messageId
                ? { ...msg, upvotes, downvotes }
                : msg
        ),
    })),

    setConnected: (connected) => set({ isConnected: connected }),

    setUsername: (name) => set({ username: name }),

    toggleOpen: () => set((state) => ({
        isOpen: !state.isOpen,
        unreadCount: !state.isOpen ? 0 : state.unreadCount,
    })),

    setOpen: (open) => set({
        isOpen: open,
        unreadCount: open ? 0 : get().unreadCount,
    }),

    resetUnread: () => set({ unreadCount: 0 }),
}));
