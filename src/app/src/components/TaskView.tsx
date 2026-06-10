import { useState } from 'react';
import { CheckCircle2, CircleOff, MapPin, MessageSquareText } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import type { Task } from '../types/Task';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
interface TaskViewProps {
    task: Task;
    taskId?: string;
    onDelete?: (taskId: string) => void;
}
function extractTime(dateStr: string | null): string {
    if (!dateStr) return "";
    if (dateStr.includes("T")) {
        return dateStr.split("T")[1]?.substring(0, 5) || "";
    }
    return dateStr;
}
function formatDuration(minutes: number): string {
    if (minutes >= 60) {
        const h = Math.floor(minutes / 60);
        const m = minutes % 60;
        return m > 0 ? `${h}h ${m}min` : `${h}h`;
    }
    return `${minutes}min`;
}
export default function TaskView({ task, taskId, onDelete }: TaskViewProps) {
    const [isNoteOpen, setIsNoteOpen] = useState(false);
    const navigate = useNavigate()
    const [taskData, setTaskData] = useState<{
        duration: number;
        difficulty: number;
        realization: boolean | null;
    }>({
        duration: typeof task?.duration === 'number' ? task.duration : 30,
        difficulty: typeof task?.difficulty === 'number' ? task.difficulty * 100 : 50,
        realization: null
    }); console.log(task);

    const handleSave = async () => {
        if (!taskId) return;

        try {
            const completed = taskData.realization === true;
            const actualDuration = completed ? taskData.duration : undefined;
            const actualDifficulty = completed ? taskData.difficulty / 100 : undefined;

            await api.rateTask(taskId, completed, actualDuration, actualDifficulty);
            setIsNoteOpen(false);
        } catch (err) {
            console.error("Failed to save task data:", err);
            alert("Failed to save: " + (err instanceof Error ? err.message : "Unknown error"));
        }
    };
    const handleModifyClick = async () => {
        if (!taskId) return;

        try {
            const fullTaskData = await api.getTask(taskId);
            const fullTask = fullTaskData.task;
            navigate("/task", { state: { task: fullTask, openMode: "modify", task_id: taskId } });
        } catch (err) {
            console.error("Failed to load task details:", err);
            const plainTask = {
                name: task.name,
                start: task.start,
                deadline: task.deadline,
                duration: task.duration,
                difficulty: task.difficulty,
                location: task.location,
                importance: task.importance,
                fixed_time: task.fixed_time,
                fixed_start: task.fixed_start,
                category: task.category,
            };
            navigate("/task", { state: { task: plainTask, openMode: "modify", task_id: taskId } });
        }
    };
    const timeDisplay = task.fixed_time
        ? extractTime(task.fixed_start)
        : `${extractTime(task.start)} → ${extractTime(task.end ?? null)}`;
    return (
        <div className="flex flex-row items-stretch">
            {/* MAIN CARD */}
            <div className={`
                bg-main p-5 flex flex-col gap-3 w-80 border border-white/5 shadow-xl z-20 transition-all duration-300
                ${isNoteOpen ? 'rounded-l-xl border-r-0' : 'rounded-xl'}
            `}>
                <h2 className="text-main text-xl font-semibold text-center mt-1 truncate">{task.name}</h2>
                <div className="flex items-center justify-between">
                    <div className="bg-sec border border-main/20 px-2.5 py-1.5 rounded-lg flex gap-1.5 items-center text-xs">
                        <MapPin size={12} className="text-main/60" />
                        <span className="text-main/80 truncate max-w-24">{task.location || "No location"}</span>
                    </div>
                    <button
                        onClick={() => setIsNoteOpen(!isNoteOpen)}
                        className={`transition-all ${isNoteOpen ? 'text-mod scale-110' : 'text-main opacity-80 hover:opacity-100'}`}
                    >
                        <MessageSquareText size={18} />
                    </button>
                </div>
                {/* TIME DISPLAY */}
                <div className="bg-main-font text-background py-2.5 rounded-xl font-bold text-base text-center tracking-wide">
                    {timeDisplay ? (
                        <span>{timeDisplay}</span>
                    ) : (
                        <span className="text-xs opacity-60">No time set</span>
                    )}
                </div>
                {/* DURATION + CATEGORIES */}
                <div className="flex flex-wrap gap-1.5 items-center">
                    {
                        task.fixed_time ??
                        <span className="px-2.5 py-1 rounded-lg bg-sec/60 border border-main/20 text-xs text-main/90 font-medium">
                            {formatDuration(task.duration)}
                        </span>

                    }
                    {task.category && task.category.slice(0, 3).map((cat, i) => (
                        <span key={i} className="px-2.5 py-1 rounded-lg bg-sec/40 border border-main/20 text-[11px] text-main/80">
                            {cat}
                        </span>
                    ))}
                    {task.category && task.category.length > 3 && (
                        <span className="text-[10px] text-main/40">+{task.category.length - 3}</span>
                    )}
                </div>
                {/* DIFFICULTY + IMPORTANCE DOTS */}
                <div className="flex gap-4 justify-center text-xs font-medium">
                    <span className="text-main/50">DIF: <span className="text-main-font">{task.difficulty.toFixed(1)}</span></span>
                    <span className="text-main/50">IMP: <span className="text-main-font">{task.importance.toFixed(1)}</span></span>
                </div>
                <div className="flex justify-between mt-1 px-1 text-xs font-medium uppercase tracking-wider">
                    <button onClick={handleModifyClick} className="text-second hover:text-main transition-colors">Modify</button>
                    <button
                        onClick={async () => {
                            if (taskId && onDelete) {
                                try {
                                    await api.deleteTask(taskId, "main_schedule");
                                    onDelete(taskId);
                                } catch (err) {
                                    console.error("Failed to delete:", err);
                                }
                            }
                        }}
                        className="text-second hover:text-del transition-colors"
                    >Delete</button>
                </div>
            </div>
            {/* INTEGRATED EXTENSION */}
            <AnimatePresence>
                {isNoteOpen && (
                    <motion.div
                        initial={{ width: 0, opacity: 0 }}
                        animate={{ width: "auto", opacity: 1 }}
                        exit={{ width: 0, opacity: 0 }}
                        transition={{ type: "spring", bounce: 0, duration: 0.4 }}
                        className="bg-main border border-white/5 border-l-0 rounded-r-xl overflow-hidden shadow-xl z-10"
                    >
                        <div className="p-5 h-full min-w-55 flex flex-col gap-4 border-l border-white/10 justify-center">
                            <div className="flex items-center justify-between gap-4">
                                <span className="text-xs text-main/90 font-medium">Realization:</span>
                                <div className="flex gap-3">
                                    <CheckCircle2
                                        size={20}
                                        style={{
                                            cursor: 'pointer',
                                            transition: 'all 0.2s ease',
                                            color: taskData.realization === true ? 'var(--main-font)' : 'var(--main-font)',
                                            opacity: taskData.realization === true ? 1 : 0.2
                                        }}
                                        onClick={() => setTaskData({ ...taskData, realization: true })}
                                    />
                                    <CircleOff
                                        size={20}
                                        style={{
                                            cursor: 'pointer',
                                            transition: 'all 0.2s ease',
                                            color: taskData.realization === false ? '#ff5f5f' : 'var(--main-font)',
                                            opacity: taskData.realization === false ? 1 : 0.2
                                        }}
                                        onClick={() => setTaskData({ ...taskData, realization: false })}
                                    />
                                </div>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="text-xs text-main/90 font-medium whitespace-nowrap">Duration:</span>
                                <div className="border-b border-second/50 flex-1 flex items-baseline gap-1" style={{ borderColor: 'rgba(255,255,255,0.1)' }}>
                                    <input
                                        type="number"
                                        value={taskData.duration}
                                        onChange={(e) => setTaskData({ ...taskData, duration: parseInt(e.target.value) || 0 })}
                                        style={{ backgroundColor: 'transparent', width: '100%', fontSize: '12px', color: 'white', outline: 'none', textAlign: 'right' }}
                                        placeholder="0"
                                    />
                                    <span className="text-[10px] text-second" style={{ opacity: 0.5 }}>min</span>
                                </div>
                            </div>
                            <div className="flex flex-col gap-1">
                                <div className="flex justify-between items-center">
                                    <span className="text-xs text-main/90 font-medium">Difficulty:</span>
                                    <span className="text-[10px]" style={{ color: 'var(--main-font)', opacity: 0.8 }}>{taskData.difficulty}%</span>
                                </div>
                                <input
                                    type="range"
                                    min="0"
                                    max="100"
                                    value={taskData.difficulty}
                                    onChange={(e) => setTaskData({ ...taskData, difficulty: parseInt(e.target.value) })}
                                    style={{ width: '100%', height: '4px', borderRadius: '8px', cursor: 'pointer', accentColor: 'var(--main-font)' }}
                                />
                            </div>
                            <button
                                onClick={handleSave}
                                className="mt-2 text-[11px] font-bold tracking-widest text-main/90 hover:text-mod active:scale-95 transition-all bg-none border-none cursor-pointer"
                            >
                                SAVE
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}