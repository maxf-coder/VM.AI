import { useState } from "react";
import { ChevronsLeftRight } from "lucide-react";

function generateDates(startDate, backCount) {
    const dates = [];
    const start = new Date(startDate);
    start.setHours(0, 0, 0, 0);

    for (let i = -backCount; i <= 6; i++) {
        const d = new Date(start);
        d.setDate(start.getDate() + i);
        dates.push(d);
    }
    return dates;
}

function formatDate(date) {
    const day = String(date.getDate()).padStart(2, "0");
    const month = String(date.getMonth() + 1).padStart(2, "0");
    return `${day}.${month}`;
}

interface MainViewDatesProps {
    selectedDate: Date;
    onDateSelect: (date: Date) => void;
}

export default function MainViewDates({ selectedDate, onDateSelect }: MainViewDatesProps) {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const [visibleBack, setVisibleBack] = useState(0);

    const dates = generateDates(today, visibleBack);

    const toggleBack = () => {
        setVisibleBack((prev) => (prev === 0 ? 3 : 0));
    };

    const isSameDay = (d1, d2) => formatDate(d1) === formatDate(d2);

    return (
        <div className="relative flex items-center gap-1 p-1.5 rounded-full w-fit backdrop-blur-md">
            {/* Directional Glass Border using main-font variable */}
            <div 
                className="absolute inset-0 rounded-full pointer-events-none"
                style={{
                    padding: "1.5px",
                    background: "linear-gradient(135deg, var(--main-font), transparent)",
                    opacity: 0.4,
                    WebkitMask: "linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)",
                    WebkitMaskComposite: "xor",
                    maskComposite: "exclude",
                }}
            />

            <button
                onClick={toggleBack}
                style={{ color: visibleBack > 0 ? "var(--main-font)" : "var(--second-font)" }}
                className={`relative z-10 p-2 transition-all duration-300 ${
                    visibleBack > 0 ? "rotate-180" : "hover:opacity-80"
                }`}
            >
                <ChevronsLeftRight size={20} strokeWidth={2.5} />
            </button>

            <div className="relative z-10 flex items-center gap-2 pr-2">
                {dates.map((date) => {
                    const isSel = isSameDay(date, selectedDate);
                    const isTdy = isSameDay(date, today);
                    const dateStr = formatDate(date);

                    return (
                        <button
                            key={dateStr}
                            onClick={() => onDateSelect(date)}
                            style={{
                                backgroundColor: isSel ? "var(--main-font)" : "var(--bg3)",
                                color: isSel ? "var(--background)" : (isTdy ? "var(--main-font)" : "var(--second-font)"),
                                borderColor: isTdy ? "var(--main-font)" : "transparent",
                            }}
                            className={`
                                px-4 py-1.5 rounded-full text-[13px] font-bold transition-all duration-200
                                border-1.5 hover:brightness-125
                            `}
                        >
                            {dateStr}
                        </button>
                    );
                })}
            </div>
        </div>
    );
}