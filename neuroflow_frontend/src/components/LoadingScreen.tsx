import { useState, useEffect } from 'react';

/**
 * Minimalist loading screen with clean white-grey-green styling.
 */
export default function LoadingScreen({ onComplete }: { onComplete: () => void }) {
    const [progress, setProgress] = useState(0);
    const [phase, setPhase] = useState('Initializing neural networks...');

    useEffect(() => {
        const phases = [
            'Initializing neural networks...',
            'Loading ST-GCN traffic model...',
            'Connecting to Bengaluru corridor...',
            'Calibrating ARAI emission factors...',
            'Starting live traffic stream...',
        ];

        let current = 0;
        const interval = setInterval(() => {
            current += Math.random() * 15 + 5;
            if (current >= 100) {
                current = 100;
                clearInterval(interval);
                setTimeout(onComplete, 500);
            }
            setProgress(current);
            setPhase(phases[Math.min(Math.floor(current / 20), phases.length - 1)] ?? 'Initializing neural networks...');
        }, 200);

        return () => clearInterval(interval);
    }, [onComplete]);

    return (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-white">
            {/* Subtle Pattern */}
            <div className="absolute inset-0 opacity-20" style={{
                backgroundImage: `radial-gradient(circle at 2px 2px, rgba(34, 197, 94, 0.1) 1px, transparent 0)`,
                backgroundSize: '32px 32px',
            }} />

            {/* Main Content */}
            <div className="relative z-10 text-center">
                {/* Logo */}
                <div className="relative inline-block mb-8">
                    <div className="w-24 h-24 rounded-2xl bg-green-500 flex items-center justify-center shadow-2xl shadow-green-500/30">
                        <span className="text-5xl">🧠</span>
                    </div>
                    {/* Subtle ring */}
                    <div className="absolute inset-0 rounded-2xl border-2 border-green-300 animate-ping opacity-30" style={{ animationDuration: '2s' }} />
                </div>

                {/* Title */}
                <h1 className="text-4xl font-extrabold mb-2 text-neutral-800">
                    NeuroFlow
                </h1>
                <p className="text-neutral-400 text-sm mb-8">BharatFlow Traffic Intelligence Platform</p>

                {/* Progress Bar */}
                <div className="w-80 mx-auto mb-4">
                    <div className="h-2 bg-neutral-200 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-green-500 rounded-full transition-all duration-300 ease-out"
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                </div>

                {/* Phase Text */}
                <p className="text-sm text-neutral-400 animate-pulse">{phase}</p>

                {/* Progress Percentage */}
                <p className="text-2xl font-bold text-neutral-300 mt-4 font-mono">
                    {Math.round(progress)}%
                </p>
            </div>

            {/* Bottom Branding */}
            <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-2 text-neutral-400 text-xs">
                <span>🇮🇳</span>
                <span>Made with ❤️ in India</span>
                <span>•</span>
                <span>Datathon 2026</span>
            </div>
        </div>
    );
}
