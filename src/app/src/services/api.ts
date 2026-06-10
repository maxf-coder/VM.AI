import type { Task, TaskResponse, ScheduleResponse, UnscheduledTask } from "../types/Task";

const BASE_URL = "http://localhost:8000/api/v1";

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    let errorMsg = `HTTP ${response.status}`;
    try {
      const errorData = await response.json();
      console.log("API Error:", errorData);
      if (errorData.detail) {
        if (typeof errorData.detail === "string") {
          errorMsg = errorData.detail;
        } else if (Array.isArray(errorData.detail)) {
          errorMsg = errorData.detail.map((e: any) => `${e.loc?.join(".")}: ${e.msg}`).join(", ");
        } else {
          errorMsg = JSON.stringify(errorData.detail);
        }
      }
    } catch (e) {
      console.error("Failed to parse error:", e);
    }
    throw new Error(errorMsg);
  }

  if (response.status === 204) {
    return {} as T;
  }

  return response.json();
}

export const api = {
  getSchedule: async (date: string): Promise<ScheduleResponse> => {
    return fetchAPI<ScheduleResponse>(`/schedule?date=${date}`);
  },

  createTask: async (task: Task, draftId?: string): Promise<TaskResponse> => {
    return fetchAPI<TaskResponse>("/tasks/", {
      method: "POST",
      body: JSON.stringify({ task, draft_id: draftId }),
    });
  },

  updateTask: async (id: string, task: Task, source: string): Promise<TaskResponse> => {
    return fetchAPI<TaskResponse>(`/tasks/${id}/update?source=${source}`, {
      method: "POST",
      body: JSON.stringify({ task }),
    });
  },

  deleteTask: async (id: string, source: string): Promise<void> => {
    return fetchAPI<void>(`/tasks/${id}?source=${source}`, {
      method: "DELETE",
    });
  },

  getTask: async (id: string): Promise<{ task_id: string; task: Task; created_at: string }> => {
    return fetchAPI(`/tasks/${id}/`);
  },

  getUnscheduledTasks: async (limit: number = 50): Promise<{ tasks: UnscheduledTask[]; total_count: number }> => {
    return fetchAPI(`/tasks/unscheduled?limit=${limit}`);
  },

  parseNLPAdd: async (prompt: string): Promise<{ draft_id: string; task: Task }> => {
    return fetchAPI("/tasks/parse/add/", {
      method: "POST",
      body: JSON.stringify({ prompt }),
    });
  },

  parseNLPModify: async (task: Task, prompt: string): Promise<{ task: Task }> => {
    return fetchAPI("/tasks/parse/modify/", {
      method: "POST",
      body: JSON.stringify({ task, prompt }),
    });
  },

  parseFromImage: async (file: File): Promise<{ prompt: string }> => {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch(`${BASE_URL}/tasks/parse/from-image`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      let errorMsg = `HTTP ${response.status}`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          errorMsg = typeof errorData.detail === "string"
            ? errorData.detail : JSON.stringify(errorData.detail);
        }
      } catch (e) { /* ignore */ }
      throw new Error(errorMsg);
    }
    return response.json();
  },

  runScheduler: async (): Promise<{
    success: boolean;
    scheduled_count: number;
    failed_count: number;
    unscheduled_remaining: string[];
    results: Array<{ success: boolean; task_id: string; message: string }>;
    execution_time_ms: number;
  }> => {
    return fetchAPI("/schedule/batch", {
      method: "POST",
    });
  },

  getProvisionalChanges: async (): Promise<{
    changes: Array<{
      task_id: string;
      task_name: string;
      change_type: string;
      new_slot_start: string;
      new_slot_end: string;
      location: string;
    }>;
    total_count: number;
  }> => {
    return fetchAPI("/provisional/changes");
  },

  resetProvisional: async (): Promise<{ success: boolean; message: string; changes_discarded: number }> => {
    return fetchAPI("/provisional/reset", {
      method: "POST",
    });
  },

  commitProvisional: async (): Promise<{ success: boolean; committed_count: number; message: string; transaction_time_ms: number }> => {
    return fetchAPI("/provisional/commit", {
      method: "POST",
    });
  },

  predictDuration: async (params: {
    difficulty: number;
    importance: number;
    scheduled_duration: number;
    category: string;
    location: string;
    fixed_time?: string;
    time_difference?: number;
  }): Promise<{ predicted_duration: number }> => {
    return fetchAPI("/tasks/predict-duration", {
      method: "POST",
      body: JSON.stringify(params),
    });
  },

  rateTask: async (id: string, completed: boolean, actualDuration?: number, actualDifficulty?: number): Promise<{
    success: boolean;
    task_id: string;
    stats_updated: boolean;
    message: string;
  }> => {
    return fetchAPI(`/tasks/${id}/rate/`, {
      method: "POST",
      body: JSON.stringify({
        completed,
        ...(completed && { actual_duration: actualDuration, actual_difficulty: actualDifficulty }),
      }),
    });
  },
};