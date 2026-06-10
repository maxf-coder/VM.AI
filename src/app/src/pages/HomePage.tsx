import { useRef, useEffect, useState } from "react";
import { motion } from "framer-motion";
import MainViewDates from "../components/MainViewDates";
import Sidebar from "../components/Sidebar";
import TaskView from "../components/TaskView";
import Background from "../components/Background";
import { api } from "../services/api";
import type { ScheduleTask } from "../types/Task";

function formatDateForAPI(date: Date): string {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

function MainView({ selectedDate, onDateSelect }: { selectedDate: Date; onDateSelect: (date: Date) => void }) {
    const containerRef = useRef<HTMLDivElement>(null);
    const [constraints, setConstraints] = useState({ left: 0, right: 0 });
    const [tasks, setTasks] = useState<ScheduleTask[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (containerRef.current) {
            const scrollWidth = containerRef.current.scrollWidth;
            const offsetWidth = containerRef.current.offsetWidth;
            setConstraints({ left: -(scrollWidth - offsetWidth), right: 0 });
        }
    }, []);

    useEffect(() => {
        const fetchSchedule = async () => {
            setLoading(true);
            try {
                const dateStr = formatDateForAPI(selectedDate);
                const response = await api.getSchedule(dateStr);
                setTasks(response.tasks);
            } catch (error) {
                console.error("Failed to fetch schedule:", error);
                setTasks([]);
            } finally {
                setLoading(false);
            }
        };
        fetchSchedule();
    }, [selectedDate]);

    return (
        <div className="flex flex-col gap-15 items-center flex-1 text-main py-18 overflow-hidden">
            <h1 className="font-bold text-[48px] mb-8">YOUR SCHEDULE</h1>
            <MainViewDates selectedDate={selectedDate} onDateSelect={onDateSelect} />

            <div className="w-full max-w-6xl mx-auto mt-4 rounded-3xl border-2 border-white/5 bg-sec/30 shadow-2xl overflow-hidden cursor-grab active:cursor-grabbing px-8 backdrop-blur-4xl">
                {loading ? (
                    <div className="flex items-center justify-center py-12 text-second-font">Loading...</div>
                ) : tasks.length === 0 ? (
                    <div className="flex items-center justify-center py-12 text-second-font">No tasks scheduled</div>
                ) : (
                    <motion.div
                        ref={containerRef}
                        drag="x"
                        dragConstraints={constraints}
                        className="flex flex-row gap-6 py-4"
                    >
                        {tasks.map((t, i) => (
                            <div key={t.task_id || i} className="shrink-0">
                                <TaskView
                                    task={{
                                        name: t.name,
                                        start: t.start,
                                        end: t.end,
                                        deadline: null,
                                        duration: t.duration ?? 60,
                                        difficulty: t.difficulty ?? 0.5,
                                        location: t.location,
                                        importance: t.importance ?? 0.5,
                                        fixed_time: t.fixed_time ?? false,
                                        fixed_start: t.fixed_start ?? null,
                                        category: t.category ?? []
                                    }}
                                    taskId={t.task_id}
                                    onDelete={(id) => setTasks(prev => prev.filter(t => t.task_id !== id))}
                                />
                            </div>
                        ))}
                    </motion.div>
                )}
            </div>
        </div>
    );
}

export default function HomePage() {
    const [selectedDate, setSelectedDate] = useState(() => {
        const d = new Date();
        d.setHours(0, 0, 0, 0);
        return d;
    });

    return (
        <div className="w-screen h-screen flex overflow-hidden">
            <Background />
            <Sidebar />
            <MainView selectedDate={selectedDate} onDateSelect={setSelectedDate} />
        </div>
    );
}