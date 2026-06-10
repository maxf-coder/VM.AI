export interface Task {
  name: string;
  start: string | null;
  end?: string | null;
  deadline: string | null;
  duration: number;
  difficulty: number;
  location: string;
  importance: number;
  fixed_time: boolean;
  fixed_start: string | null;
  category: string[];
}

export function createDefaultTask(): Task {
  return {
    name: "",
    start: null,
    deadline: null,
    duration: 30,
    difficulty: 0.5,
    location: "",
    importance: 0.5,
    fixed_time: false,
    fixed_start: null,
    category: [],
  };
}

export interface TaskResponse {
  success: boolean;
  task_id: string;
  status: string;
  message?: string;
}

export interface ScheduleTask {
  task_id: string;
  name: string;
  start: string;
  end: string;
  location: string;
  rated: boolean;
  fixed_time: boolean;
  fixed_start: string | null;
  duration: number;
  category: string[];
  difficulty: number;
  importance: number; 
}

export interface ScheduleResponse {
  date: string;
  tasks: ScheduleTask[];
}

export interface UnscheduledTask {
  task_id: string;
  task: Task;
  created_at: string;
}