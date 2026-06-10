import React, { useEffect, useRef, useState } from "react";
import { ArrowUp, ImagePlus } from "lucide-react";
import { api } from "../services/api";
import type { Task } from "../types/Task";

interface NLPViewProps {
  mode: "add" | "modify";
  initialTask?: Task;
  onParsedTask?: (task: Task) => void;
}

export default function NLPView({ mode, initialTask, onParsedTask }: NLPViewProps) {
    const [modeText, setModeText] = useState("Add");
    const [inputText, setInputText] = useState("");
    const [loading, setLoading] = useState(false);
    const [imageLoading, setImageLoading] = useState(false);
    const [error, setError] = useState("");
    const [parsedResult, setParsedResult] = useState<Task | null>(initialTask ?? null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (mode === "add") setModeText("Add");
        else setModeText("Modify");
    }, [mode]);

    useEffect(() => {
        setParsedResult(initialTask ?? null);
    }, [initialTask]);

    const handleSubmit = async () => {
        if (!inputText.trim()) return;
        
        setLoading(true);
        setError("");
        
        try {
            let result;
            if (mode === "modify" && parsedResult) {
                result = await api.parseNLPModify(parsedResult, inputText);
                const updatedTask = { ...parsedResult, ...result.task };
                setParsedResult(updatedTask);
                if (onParsedTask) {
                    onParsedTask(updatedTask);
                }
                console.log(updatedTask);
            } else {
                result = await api.parseNLPAdd(inputText);
                setParsedResult(result.task);
                if (onParsedTask) {
                    onParsedTask(result.task);
                }
                console.log(result.task);
            }
        } catch (err: any) {
            setError(err.message || "Failed to parse task");
        } finally {
            setLoading(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    const handleImageSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        setImageLoading(true);
        setError("");
        try {
            const result = await api.parseFromImage(file);
            setInputText(result.prompt);
        } catch (err: any) {
            setError(err.message || "Failed to parse image");
        } finally {
            setImageLoading(false);
            if (fileInputRef.current) fileInputRef.current.value = "";
        }
    };

    return (
        <div className="flex flex-col items-center justify-between h-full py-24 px-8 bg-main">
            <div className="flex flex-col items-center gap-2">
                    <h1 className="text-main font-bold text-6xl tracking-tight">
                        {modeText}
                    </h1>
            </div>

            <div className="w-full flex-1 max-h-125 mt-12 mb-8 rounded-[40px] border border-white/5 bg-sec/10 relative flex flex-col p-4 shadow-inner">
                <div className="flex-1 overflow-y-auto p-4 text-main/40 font-light italic">
                    {parsedResult ? (
                        <div className="text-main font-normal">
                            <p><strong>Name:</strong> {parsedResult.name}</p>
                            <p><strong>Duration:</strong> {parsedResult.duration} min</p>
                            <p><strong>Difficulty:</strong> {parsedResult.difficulty}</p>
                            <p><strong>Category:</strong> {parsedResult.category?.join(", ")}</p>
                            <p><strong>Location:</strong> {parsedResult.location}</p>
                            <p><strong>Importance:</strong> {parsedResult.importance}</p>
                        </div>
                    ) : (
                        "Start describing your task..."
                    )}
                </div>
                
                {error && (
                    <div className="text-red-400 text-sm p-2">{error}</div>
                )}

                <div className="relative mt-auto">
                    {mode === "add" && (
                        <input
                            type="file"
                            accept="image/jpeg,image/png,image/webp"
                            ref={fileInputRef}
                            onChange={handleImageSelect}
                            className="hidden"
                        />
                    )}
                    <input
                        type="text"
                        placeholder="Add a new task"
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        onKeyDown={handleKeyPress}
                        disabled={loading}
                        className={`w-full bg-main border border-white/10 rounded-2xl py-4 pr-14 text-main placeholder:text-main/20 outline-none focus:border-main-font/40 transition-colors ${mode === "add" ? "pl-14" : "px-6"}`}
                    />
                    {mode === "add" && (
                        <button
                            type="button"
                            onClick={() => fileInputRef.current?.click()}
                            disabled={imageLoading}
                            className="absolute left-3 top-1/2 -translate-y-1/2 bg-main-font/10 p-2 rounded-xl hover:bg-main-font/20 active:scale-95 transition-all disabled:opacity-50"
                        >
                            {imageLoading ? (
                                <svg className="animate-spin" width="20" height="20" viewBox="0 0 24 24" fill="none">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                </svg>
                            ) : (
                                <ImagePlus size={20} className="text-main-font" />
                            )}
                        </button>
                    )}
                    <button 
                        onClick={handleSubmit}
                        disabled={loading || !inputText.trim()}
                        className="absolute right-3 top-1/2 -translate-y-1/2 bg-main-font p-2 rounded-xl hover:scale-105 active:scale-95 transition-all shadow-lg disabled:opacity-50"
                    >
                        <ArrowUp size={20} className="text-background" />
                    </button>
                </div>
            </div>

            <button 
                onClick={handleSubmit}
                disabled={loading || !inputText.trim()}
                className="w-full bg-main-font text-background font-bold py-5 rounded-[20px] text-lg hover:opacity-90 active:scale-[0.98] transition-all shadow-xl uppercase tracking-widest disabled:opacity-50"
            >
                {loading ? "Parsing..." : "Submit request"}
            </button>
        </div>
    );
}