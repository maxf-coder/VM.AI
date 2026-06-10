import { Calendar, Calendar1, BarChart, LucideIcon } from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";

interface FieldProps {
  data: {
    id: number;
    name: string;
    icon: "schedule" | "task" | "stats";
    path: string;
  };
  isSelected: boolean;
  onTap: () => void;
}

function Field({ data, isSelected, onTap }: FieldProps) {
    const iconMap: Record<string, LucideIcon> = {
        schedule: Calendar,
        task: Calendar1,
        stats: BarChart,
    };

    const IconComp = iconMap[data.icon];

    return (
        <div
            className="relative flex flex-row items-center justify-end gap-3 text-main cursor-pointer py-1"
            onClick={onTap}
        >
            {/* The Sliding Indicator */}
            <AnimatePresence>
                {isSelected && (
                    <motion.div
                        layoutId="sidebar-accent"
                        className="absolute -right-4 w-1 h-full bg-mod shadow-[0_0_15px_rgba(255,214,102,0.5)]"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    />
                )}
            </AnimatePresence>

            <motion.div
                animate={{ x: isSelected ? -20 : 0 }}
                transition={{ type: "spring", stiffness: 300, damping: 30 }}
                className={`flex items-center gap-3 ${isSelected ? 'opacity-100' : 'opacity-40 hover:opacity-100'}`}
            >
                <h1 className="text-[16px] font-light uppercase tracking-widest">{data.name}</h1>
                <IconComp size={22} strokeWidth={1.5} />
            </motion.div>
        </div>
    );
}

export default function Sidebar() {
    const navigate = useNavigate();
    const location = useLocation();

    const fields: { id: number; name: string; icon: "schedule" | "task" | "stats"; path: string; }[] = [
        { id: 1, name: "Schedule", icon: "schedule", path: "/" },
        { id: 2, name: "Add a task", icon: "task", path: "/task" },
        { id: 3, name: "Pending changes", icon: "stats", path: "/pending" },
    ];

    return (
        <div className="w-64 bg-main h-full flex flex-col p-6 gap-8 border-r border-white/5">
            <div className="py-4">
                <h1 className="text-main text-[62px] font-bold leading-none tracking-tighter">VM.AI</h1>
                <h2 className="text-second font-medium text-[18px] uppercase tracking-[0.2em] mt-1">
                    set your day
                </h2>
            </div>

            <div className="flex flex-col gap-8 mt-10">
                {fields.map((field) => (
                    <Field
                        key={field.id}
                        data={field}
                        isSelected={location.pathname === field.path}
                        onTap={() => navigate(field.path)}
                    />
                ))}
            </div>
        </div>
    );
}