/**
 * API client for FastAPI backend
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

// Types
export interface User {
  id: number;
  email: string;
  username: string;
}

export interface Task {
  id: number;
  name: string;
  display_name: string;
  vlmeval_data: string;
  description: string | null;
  primary_metric_key: string;
  dataset_version: string | null;
  num_examples: number | null;
  paper_url: string | null;
  run_count?: number;
}

export interface Model {
  id: number;
  name: string;
  display_name: string;
  vlmeval_model: string;
  model_type: string;
  params_b: number | null;
}

export interface EvalRun {
  id: number;
  task_id: number;
  model_id: number;
  status: "QUEUED" | "RUNNING" | "COMPLETE" | "FAILED";
  metrics: Record<string, any> | null;
  artifacts_dir: string | null;
  command: string | null;
  git_commit: string | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  task_name?: string;
  model_name?: string;
  model_display_name?: string;
  primary_metric?: number;
  duration_seconds?: number;
}

export interface LeaderboardEntry {
  model_name: string;
  model_display_name: string;
  primary_metric: number | null;
  run_id: number;
  run_date: string;
  git_commit: string | null;
  status: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  image_url?: string;
}

export interface Image {
  id: number;
  filename: string;
  url: string;
  is_public: boolean;
  created_at: string;
  user_id: number;
}

// Token management
let accessToken: string | null = null;

export function setAccessToken(token: string | null) {
  accessToken = token;
  if (token) {
    localStorage.setItem("access_token", token);
  } else {
    localStorage.removeItem("access_token");
  }
}

export function getAccessToken(): string | null {
  if (accessToken) return accessToken;
  if (typeof window !== "undefined") {
    accessToken = localStorage.getItem("access_token");
  }
  return accessToken;
}


// Decode JWT to get user info (client-side)
function decodeJWT(token: string): any {
  try {
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
    return JSON.parse(jsonPayload);
  } catch {
    return null;
  }
}




// Base fetch with auth
async function fetchWithAuth(
  endpoint: string,
  options: RequestInit = {}
): Promise<Response> {
  const token = getAccessToken();
  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...options.headers,
  };

  if (token) {
    (headers as Record<string, string>)["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  // Handle 401 - clear token and redirect
  if (response.status === 401) {
    setAccessToken(null);
    if (typeof window !== "undefined" && !window.location.pathname.includes('/login')) {
      window.location.href = "/login";
    }
  }

  return response;
}

// API functions
export const api = {
  // Auth - Updated to match your FastAPI endpoints
  async login(email: string, password: string): Promise<{ access_token: string; user: User }> {
    // Your endpoint uses OAuth2PasswordRequestForm which expects form data
    const formData = new URLSearchParams();
    formData.append('username', email); // OAuth2 form uses 'username' field
    formData.append('password', password);

    const response = await fetch(`${API_BASE}/auth/token`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Login failed");
    }

    const data = await response.json();
    setAccessToken(data.access_token);

    // Decode JWT to get user info since you don't have a /me endpoint
    const decoded = decodeJWT(data.access_token);
    const user: User = {
      id: decoded?.user_id || 0,
      email: decoded?.sub || email,
    };

    return { access_token: data.access_token, user };
  },

  async register(email: string, username: string, password: string): Promise<{ access_token: string; user: User }> {
    // First, sign up
    const response = await fetch(`${API_BASE}/auth/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Registration failed");
    }

    const signupData = await response.json();

    // Then automatically log in to get the token
    const loginResult = await api.login(email, password);
    return loginResult;
  },

  async getCurrentUser(): Promise<User> {
    // Since you don't have a /me endpoint, decode the token
    const token = getAccessToken();
    if (!token) {
      throw new Error("Not authenticated");
    }

    const decoded = decodeJWT(token);
    if (!decoded) {
      throw new Error("Invalid token");
    }

    // Check if token is expired
    if (decoded.exp && decoded.exp * 1000 < Date.now()) {
      setAccessToken(null);
      throw new Error("Token expired");
    }

    return {
      id: decoded.user_id || 0,
      email: decoded.sub || "",
    };
  },

  logout() {
    setAccessToken(null);
  },



  // Tasks
  async getTasks(): Promise<Task[]> {
    const response = await fetchWithAuth("/api/evals/tasks");
    if (!response.ok) throw new Error("Failed to fetch tasks");
    return response.json();
  },

  async getTask(name: string): Promise<Task> {
    const response = await fetchWithAuth(`/api/evals/tasks/${name}`);
    if (!response.ok) throw new Error("Failed to fetch task");
    return response.json();
  },


  async getAvailableMetrics(taskName: string): Promise<string[]> {
    const response = await fetchWithAuth(`/api/evals/tasks/${taskName}/metrics`);
    if (!response.ok) throw new Error("Failed to fetch metrics");
    return response.json();
  },

  async getLeaderboard(taskName: string, metric?: string): Promise<LeaderboardEntry[]> {
    const params = new URLSearchParams();
    if (metric) params.set("metric", metric);
    const response = await fetchWithAuth(`/api/evals/tasks/${taskName}/leaderboard?${params}`);
    if (!response.ok) throw new Error("Failed to fetch leaderboard");
    return response.json();
  },


  async getTaskRuns(taskName: string, status?: string): Promise<EvalRun[]> {
    const params = new URLSearchParams();
    if (status) params.set("status", status);
    const response = await fetchWithAuth(`/api/evals/tasks/${taskName}/runs?${params}`);
    if (!response.ok) throw new Error("Failed to fetch runs");
    return response.json();
  },

  async triggerEval(taskName: string, modelName: string): Promise<EvalRun> {
    const response = await fetchWithAuth(`/api/evals/tasks/${taskName}/runs`, {
      method: "POST",
      body: JSON.stringify({ model_name: modelName }),
    });
    if (!response.ok) throw new Error("Failed to trigger evaluation");
    return response.json();
  },

  // Models
  async getModels(): Promise<Model[]> {
    const response = await fetchWithAuth("/api/evals/models");
    if (!response.ok) throw new Error("Failed to fetch models");
    return response.json();
  },

  // Runs
  async getRun(runId: number): Promise<EvalRun> {
    const response = await fetchWithAuth(`/api/evals/runs/${runId}`);
    if (!response.ok) throw new Error("Failed to fetch run");
    return response.json();
  },

  // Chat
  async sendMessage(
    messages: ChatMessage[],
    imageUrl?: string
  ): Promise<ReadableStream<Uint8Array>> {
    const response = await fetchWithAuth("/api/chat", {
      method: "POST",
      body: JSON.stringify({ messages, image_url: imageUrl }),
    });
    if (!response.ok) throw new Error("Chat request failed");
    return response.body!;
  },

  // Images
  async getImages(options?: {
    public?: boolean,
    limit?: number; 
    offset?: number;
  }): Promise<Image[]> {
    const params = new URLSearchParams();
 
    if (options?.public !== undefined) params.set("public", String(options.public));
    if (options?.limit !== undefined) params.set("limit", String(options.limit));
    if (options?.offset !== undefined) params.set("offset", String(options.offset));

    console.log("Fetching images with params:", params.toString());  // Add this

    
    const response = await fetchWithAuth(`/images/me?${params}`);
    //const images = await response.json();    
    if (!response.ok) throw new Error("Failed to fetch images");
    return response.json();
    //return images;
  },

  async uploadImage(file: File, isPublic: boolean): Promise<Image> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("is_public", String(isPublic));

    const token = getAccessToken();
    //const response = await fetch(`${API_BASE}/api/images/upload`, {
    const response = await fetch(`/images/ingest_upload`, {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      body: formData,
    });
    if (!response.ok) throw new Error("Failed to upload image");
    return response.json();
  },

  async toggleImageVisibility(imageId: number, isPublic: boolean): Promise<Image> {
    //const response = await fetchWithAuth(`/api/images/${imageId}`, {
    const response = await fetchWithAuth(`/images/${imageId}`, {
      method: "PATCH",
      body: JSON.stringify({ is_public: isPublic }),
    });
    if (!response.ok) throw new Error("Failed to update image");
    return response.json();
  },
};
