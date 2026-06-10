import { useState, useEffect } from "react";
import { MapPin } from "lucide-react";
import { useNavigate } from "react-router-dom";
import Sidebar from "../components/Sidebar";
import Background from "../components/Background";
import TaskView from "../components/TaskView";
import type { Task, UnscheduledTask } from "../types/Task";
import { api } from "../services/api";

interface ProvisionalChange {
    task_id: string;
    task_name: string;
    change_type: string;
    new_slot_start: string;
    new_slot_end: string;
    location: string;
}

function ScheduleChangesView({ changes, loading, onModify, onDelete }: {
    changes: ProvisionalChange[];
    loading: boolean;
    onModify: (taskId: string) => void;
    onDelete: (taskId: string) => void;
}) {
    if (loading) {
        return <div className="text-second-font">Loading...</div>;
    }
    if (changes.length === 0) {
        return (
            <div className="w-full grid grid-cols-4 auto-rows-fr gap-4">
                <div className="text-second-font p-4">No pending schedule changes</div>
            </div>
        );
    }
    return (
        <div className="w-full grid grid-cols-4 auto-rows-fr gap-4">
            {changes.map((change) => {
                const changeDate = change.new_slot_start ? change.new_slot_start.split("T")[0] : "";
                const startTime = change.new_slot_start ? change.new_slot_start.split("T")[1]?.substring(0, 5) : "";
                const endTime = change.new_slot_end ? change.new_slot_end.split("T")[1]?.substring(0, 5) : "";
                return (
                    <div key={change.task_id} className="flex flex-col gap-3 p-4 border border-white/20 rounded-xl bg-white/5">
                        <div className="flex flex-row items-center justify-between">
                            <span className="text-lg text-main-font font-medium truncate">{change.task_name}</span>
                        </div>
                        <div className="text-sm text-second-font font-medium">
                            {changeDate} &middot; {startTime} - {endTime}
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                            <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-sec/40 border border-white/10">
                                <MapPin className="w-3 h-3 text-second-font" />
                                <span className="text-xs text-second-font">{change.location}</span>
                            </div>
                            <div className="px-2 py-1 rounded-md bg-sec/40 border border-white/10">
                                <span className="text-xs text-second-font capitalize">{change.change_type}</span>
                            </div>
                        </div>
                        <div className="flex justify-between mt-1 px-1 text-[9px] font-medium uppercase tracking-tighter">
                            <button onClick={() => onModify(change.task_id)} className="text-second hover:text-main transition-colors">Modify</button>
                            <button onClick={() => onDelete(change.task_id)} className="text-second hover:text-del transition-colors">Delete</button>
                        </div>
                    </div>
                );
            })}
        </div>
    );
}

function UnscheduledTaskView({ task, onDelete }: { task: UnscheduledTask; onDelete: (id: string) => void }) {
    const navigate = useNavigate();
    const t = task.task;
    const startTime = t.start ? t.start.split("T")[1]?.substring(0, 5) : "";
    const deadlineDisplay = t.deadline ? t.deadline.split("T")[0] : "";
    const durationDisplay = t.duration ? `${t.duration}min` : "";

    return (
        <div className="flex flex-col gap-2 p-4 border border-white/20 rounded-xl bg-white/5">
            <div className="flex flex-row items-center justify-between">
                <span className="text-lg text-main-font font-medium truncate flex-1">{t.name}</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
                {startTime && (
                    <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-sec/40 border border-white/10">
                        <span className="text-xs text-second-font font-medium">{startTime}</span>
                    </div>
                )}
                {durationDisplay && (
                    <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-sec/40 border border-white/10">
                        <span className="text-xs text-second-font">{durationDisplay}</span>
                    </div>
                )}
                {deadlineDisplay && (
                    <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-sec/40 border border-white/10">
                        <span className="text-xs text-second-font">{deadlineDisplay}</span>
                    </div>
                )}
                {t.location && (
                    <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-sec/40 border border-white/10">
                        <MapPin className="w-3 h-3 text-second-font" />
                        <span className="text-xs text-second-font">{t.location}</span>
                    </div>
                )}
            </div>
            {t.category && t.category.length > 0 && (
                <div className="flex flex-wrap gap-1">
                    {t.category.map((cat, i) => (
                        <span key={i} className="px-2 py-0.5 rounded-md bg-main-font/10 border border-main-font/20">
                            <span className="text-[10px] text-main-font/80">{cat}</span>
                        </span>
                    ))}
                </div>
            )}
            <div className="flex justify-between mt-1 px-1 text-[9px] font-medium uppercase tracking-tighter">
                <button onClick={() => navigate("/task", { state: { task: t, task_id: task.task_id, openMode: "modify", source: "unscheduled" } })} className="text-second hover:text-main transition-colors">Modify</button>
                <button onClick={() => onDelete(task.task_id)} className="text-second hover:text-del transition-colors">Delete</button>
            </div>
        </div>
    );
}

function UnscheduledChangesView({ tasks, loading, onDelete }: { tasks: UnscheduledTask[]; loading: boolean; onDelete: (id: string) => void }) {
    if (loading) {
        return <div className="text-second-font">Loading...</div>;
    }
    if (tasks.length === 0) {
        return (
            <div className="w-full grid grid-cols-4 auto-rows-fr gap-4">
                <div className="text-second-font p-4">No unscheduled tasks</div>
            </div>
        );
    }
    return (
        <div className="w-full grid grid-cols-4 auto-rows-fr gap-4">
            {tasks.map((t) => (
                <UnscheduledTaskView key={t.task_id} task={t} onDelete={onDelete} />
            ))}
        </div>
    );
}

export default function PendingTasksPage() {
    const navigate = useNavigate();
    const [activeView, setActiveView] = useState("unscheduled");
    const [unscheduledTasks, setUnscheduledTasks] = useState<UnscheduledTask[]>([]);
    const [provisionalChanges, setProvisionalChanges] = useState<ProvisionalChange[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [warnings, setWarnings] = useState<{ task_id: string; task_name: string; message: string }[]>([]);

    const fetchUnscheduled = async () => {
        setLoading(true);
        try {
            const response = await api.getUnscheduledTasks();
            setUnscheduledTasks(response.tasks);
            setError("");
        } catch (err) {
            setError("Failed to fetch tasks");
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const fetchProvisionalChanges = async () => {
        setLoading(true);
        try {
            const response = await api.getProvisionalChanges();
            setProvisionalChanges(response.changes);
            setError("");
        } catch (err) {
            setError("Failed to fetch changes");
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (activeView === "unscheduled") {
            fetchUnscheduled();
        } else {
            fetchProvisionalChanges();
        }
    }, [activeView]);

    const handleDelete = async (taskId: string) => {
        try {
            await api.deleteTask(taskId, "unscheduled");
            fetchUnscheduled();
        } catch (err) {
            console.error("Failed to delete task:", err);
        }
    };

    const handleSchedule = async () => {
        setLoading(true);
        setWarnings([]);
        try {
            const result = await api.runScheduler(); console.log(result);
            const failed = result.results.filter(r => !r.success);
            if (failed.length > 0) {
                const nameMap = new Map(unscheduledTasks.map(t => [t.task_id, t.task.name]));
                setWarnings(failed.map(f => ({
                    task_id: f.task_id,
                    task_name: nameMap.get(f.task_id) || f.task_id.slice(0, 8),
                    message: f.message,
                })));
            }
            setActiveView("schedule");
            await fetchProvisionalChanges();
        } catch (err) {
            console.error("Failed to run scheduler:", err);
            alert("Failed to run scheduler: " + (err instanceof Error ? err.message : "Unknown error"));
        } finally {
            setLoading(false);
        }
    };

    const handleProvisionalDelete = async (taskId: string) => {
        try {
            await api.deleteTask(taskId, "provisional");
            fetchProvisionalChanges();
        } catch (err) {
            console.error("Failed to delete provisional change:", err);
        }
    };

    const handleProvisionalModify = async (taskId: string) => {
        try {
            const fullTaskData = await api.getTask(taskId);
            navigate("/task", { state: { task: fullTaskData.task, task_id: taskId, openMode: "modify", source: "provisional" } });
        } catch (err) {
            console.error("Failed to load task details:", err);
        }
    };

    const handleReset = async () => {
        setLoading(true);
        try {
            await api.resetProvisional();
            const response = await api.getProvisionalChanges();
            setProvisionalChanges(response.changes);
        } catch (err) {
            console.error("Failed to reset:", err);
        } finally {
            setLoading(false);
        }
    };

    const handleCommit = async () => {
        try {
            await api.commitProvisional();
            setProvisionalChanges([]);
            setActiveView("unscheduled");
            fetchUnscheduled();
        } catch (err) {
            console.error("Failed to commit:", err);
        }
    };

    return (
        <div className="w-screen h-screen flex overflow-hidden">
            <Background />
            <Sidebar />

            <div className="flex-1 flex items-center justify-center p-12">
                <div className="w-275 h-175 rounded-[40px] border border-white/5 bg-sec/30 shadow-2xl backdrop-blur-xl flex flex-col overflow-hidden">
                    <div className="p-6 border-b border-white/5">
                        <div className="flex flex-row items-center justify-between">
                            <div className="flex flex-row gap-3">
                                <button
                                    onClick={() => setActiveView("schedule")}
                                    className={`
                                        px-6 py-4 rounded-xl text-sm font-medium tracking-wide transition-all duration-200
                                        border backdrop-blur-md
                                        ${activeView === "schedule"
                                            ? "border-main-font/40 bg-main-font/10 text-main-font"
                                            : "border-white/10 bg-sec/20 text-main/60 hover:text-main hover:border-white/20"
                                        }
                                    `}
                                >
                                    Schedule Changes
                                </button>

                                <button
                                    onClick={() => setActiveView("unscheduled")}
                                    className={`
                                        px-6 py-4 rounded-xl text-sm font-medium tracking-wide transition-all duration-200
                                        border backdrop-blur-md
                                        ${activeView === "unscheduled"
                                            ? "border-main-font/40 bg-main-font/10 text-main-font"
                                            : "border-white/10 bg-sec/20 text-main/60 hover:text-main hover:border-white/20"
                                        }
                                    `}
                                >
                                    Unscheduled Changes
                                </button>
                            </div>

                            <div className="flex flex-row gap-3">
                                {activeView === "schedule" ? (
                                    <>
                                        <button onClick={handleReset} disabled={loading} className="border border-main-font/30 text-main-font py-4 px-6 rounded-xl text-sm font-medium tracking-wide hover:bg-main-font/10 transition-all disabled:opacity-50">
                                            {loading ? "Resetting..." : "Reset to main schedule"}
                                        </button>
                                        <button onClick={handleCommit} disabled={loading} className="border border-main-font/30 text-main-font py-4 px-6 rounded-xl text-sm font-medium tracking-wide hover:bg-main-font/10 transition-all disabled:opacity-50">
                                            {loading ? "Submitting..." : "Submit changes"}
                                        </button>
                                    </>
                                ) : (
                                    <button onClick={handleSchedule} disabled={loading} className="border border-main-font/30 text-main-font py-4 px-6 rounded-xl text-sm font-medium tracking-wide hover:bg-main-font/10 transition-all">
                                        {loading ? "Scheduling..." : "Schedule the tasks"}
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>

                    <div className="flex-1 p-8 relative">
                        {error && <div className="text-red-500 mb-4">{error}</div>}
                        {warnings.length > 0 && (
                            <div className="mb-4 p-4 rounded-xl border border-yellow-500/40 bg-yellow-500/10 text-yellow-200 text-sm">
                                <div className="flex items-start justify-between gap-4">
                                    <div className="flex flex-col gap-1">
                                        <span className="font-semibold">{warnings.length} task{warnings.length > 1 ? "s" : ""} couldn't be scheduled:</span>
                                        {warnings.map(w => (
                                            <div key={w.task_id} className="flex gap-2">
                                                <span className="font-medium">• {w.task_name}:</span>
                                                <span>{w.message}</span>
                                            </div>
                                        ))}
                                    </div>
                                    <button onClick={() => setWarnings([])} className="text-yellow-300/60 hover:text-yellow-200 shrink-0 text-lg leading-none">&times;</button>
                                </div>
                            </div>
                        )}
                        {activeView === "schedule" ? (
                            <ScheduleChangesView changes={provisionalChanges} loading={loading} onModify={handleProvisionalModify} onDelete={handleProvisionalDelete} />
                        ) : (
                            <UnscheduledChangesView tasks={unscheduledTasks} loading={loading} onDelete={handleDelete} />
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}