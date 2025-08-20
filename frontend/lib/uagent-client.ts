/**
 * uAgent Client for direct integration with AI Data Science Platform agents
 * 
 * This client handles communication with 6 uAgents running on ports 8004-8009:
 * - 8004: Data Cleaning Agent 
 * - 8005: Data Loader Agent
 * - 8006: Data Visualization Agent  
 * - 8007: Feature Engineering Agent
 * - 8008: H2O ML Training Agent
 * - 8009: ML Prediction Agent
 */

export type SessionId = string;

// Base response types
export interface SessionResponse {
  success: boolean;
  message: string;
  session_id: SessionId;
  execution_time_seconds?: number;
  error?: string;
}

export interface HealthResponse {
  status: string;
  agent_status: string;
}

// Operation parameter types
export interface LoadFileParams {
  filename: string;
  file_content: string; // Base64 encoded
  user_instructions?: string;
}

export interface LoadDirectoryParams {
  directory_path: string;
  user_instructions?: string;
}

export interface CleanDataParams {
  session_id?: string;
  filename?: string;
  file_content?: string; // Base64 encoded CSV
  user_instructions?: string;
  advanced_options?: {
    remove_duplicates?: boolean;
    handle_missing?: 'drop' | 'fill' | 'interpolate';
    normalize_columns?: boolean;
    detect_outliers?: boolean;
  };
}

export interface VizParams {
  session_id?: string;
  filename?: string; 
  file_content?: string; // Base64 encoded CSV
  chart_type?: string;
  x_column?: string;
  y_column?: string;
  user_instructions?: string;
}

export interface FeatureParams {
  session_id?: string;
  filename?: string;
  file_content?: string; // Base64 encoded CSV
  feature_goals?: string;
  constraints?: string;
  user_instructions?: string;
}

export interface TrainingParams {
  session_id?: string;
  filename?: string;
  file_content?: string; // Base64 encoded CSV
  target_column: string;
  time_budget_seconds?: number;
  user_instructions?: string;
}

export interface PredictionParams {
  model_session_id?: string;
  model_path?: string;
  input_data: Record<string, unknown> | Array<Record<string, unknown>>;
  analysis_options?: {
    include_probabilities?: boolean;
    include_feature_importance?: boolean;
  };
}

// Result response types  
export interface DataResponse {
  data?: {
    records: Array<Record<string, unknown>>;
    columns: string[];
  };
}

export interface CodeResponse {
  code?: string;
  generated_code?: string; // Backend uses this field name
}

export interface ChartResponse {
  figure?: unknown; // Plotly JSON
}

export interface LeaderboardResponse {
  leaderboard?: Array<Record<string, unknown>>;
}

export interface LogsResponse {
  logs?: string[];
  messages?: string[];
}

export interface RecommendationsResponse {
  recommendations?: string[];
  workflow_summary?: string;
  cleaning_steps?: string[];
  visualization_steps?: string[];
  engineering_steps?: string[];
  ml_steps?: string[];
}

export interface AnalysisResponse {
  analysis?: string;
  model_analysis?: Record<string, unknown>;
  prediction_results?: Record<string, unknown>;
  batch_results?: Array<Record<string, unknown>>;
}

// Agent types
export type AgentType = 'loading' | 'cleaning' | 'visualization' | 'engineering' | 'training' | 'prediction';

// Base URLs for each agent
const AGENT_BASE_URLS: Record<AgentType, string> = {
  loading: 'http://127.0.0.1:8005',
  cleaning: 'http://127.0.0.1:8004', 
  visualization: 'http://127.0.0.1:8006',
  engineering: 'http://127.0.0.1:8007',
  training: 'http://127.0.0.1:8008',
  prediction: 'http://127.0.0.1:8009',
};

export class UAgentClient {
  private baseUrl: string;
  
  constructor(private agentType: AgentType) {
    this.baseUrl = AGENT_BASE_URLS[agentType];
  }

  // Health check
  async checkHealth(): Promise<HealthResponse> {
    const response = await fetch(`${this.baseUrl}/health`, {
      method: 'GET',
    });
    
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }
    
    return response.json();
  }

  // Generic request helper
  private async request<T>(endpoint: string, data?: unknown): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: data ? JSON.stringify(data) : undefined,
    });
    
    if (!response.ok) {
      throw new Error(`Request failed: ${response.statusText}`);
    }
    
    return response.json();
  }

  private async getSessionResult<T>(endpoint: string): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'GET',
    });
    
    if (!response.ok) {
      throw new Error(`Session result fetch failed: ${response.statusText}`);
    }
    
    return response.json();
  }

  // Data Loader operations (8005)
  async loadFile(params: LoadFileParams): Promise<SessionResponse> {
    return this.request<SessionResponse>('/load-file', params);
  }

  async loadDirectory(params: LoadDirectoryParams): Promise<SessionResponse> {
    return this.request<SessionResponse>('/load-directory', params);
  }

  // Data Cleaning operations (8004)
  async cleanData(params: CleanDataParams): Promise<SessionResponse> {
    if (params.session_id) {
      return this.request<SessionResponse>('/clean-data', params);
    } else {
      return this.request<SessionResponse>('/clean-csv', params);
    }
  }

  // Visualization operations (8006)  
  async createChart(params: VizParams): Promise<SessionResponse> {
    if (params.session_id) {
      return this.request<SessionResponse>('/create-chart', params);
    } else {
      return this.request<SessionResponse>('/create-chart-csv', params);
    }
  }

  // Feature Engineering operations (8007)
  async engineerFeatures(params: FeatureParams): Promise<SessionResponse> {
    if (params.session_id) {
      return this.request<SessionResponse>('/engineer-features', params);
    } else {
      return this.request<SessionResponse>('/engineer-features-csv', params);
    }
  }

  // ML Training operations (8008)
  async trainModel(params: TrainingParams): Promise<SessionResponse> {
    if (params.session_id) {
      return this.request<SessionResponse>('/train-model', params);
    } else {
      return this.request<SessionResponse>('/train-model-csv', params);
    }
  }

  // ML Prediction operations (8009)
  async predictSingle(params: PredictionParams): Promise<SessionResponse> {
    return this.request<SessionResponse>('/predict-single', params);
  }

  async predictBatch(params: PredictionParams): Promise<SessionResponse> {
    return this.request<SessionResponse>('/predict-batch', params);
  }

  async analyzeModel(params: { model_session_id?: string; model_path?: string }): Promise<SessionResponse> {
    return this.request<SessionResponse>('/analyze-model', params);
  }

  // Session result getters  
  async getSessionData(sessionId: string): Promise<DataResponse> {
    // Use the POST /get-artifacts endpoint instead of GET session endpoint
    if (this.agentType === 'loading') {
      return this.request<DataResponse>('/get-artifacts', { session_id: sessionId });
    }
    return this.getSessionResult<DataResponse>(`/session/${sessionId}/data`);
  }

  async getSessionCode(sessionId: string): Promise<CodeResponse> {
    switch (this.agentType) {
      case 'cleaning':
        const response = await this.request<{success: boolean, generated_code?: string, error?: string}>('/get-cleaning-function', { session_id: sessionId });
        return { 
          code: response.generated_code,
          generated_code: response.generated_code
        };
      case 'visualization':
        return this.request<CodeResponse>('/get-visualization-function', { session_id: sessionId });
      case 'engineering':
        return this.request<CodeResponse>('/get-engineering-function', { session_id: sessionId });
      case 'training':
        return this.request<CodeResponse>('/get-training-function', { session_id: sessionId });
      default:
        throw new Error(`Code not available for agent type: ${this.agentType}`);
    }
  }

  async getSessionChart(sessionId: string): Promise<ChartResponse> {
    if (this.agentType !== 'visualization') {
      throw new Error('Charts only available for visualization agent');
    }
    return this.request<ChartResponse>('/get-plotly-graph', { session_id: sessionId });
  }

  async getSessionLeaderboard(sessionId: string): Promise<LeaderboardResponse> {
    if (this.agentType !== 'training') {
      throw new Error('Leaderboard only available for training agent');
    }
    return this.request<LeaderboardResponse>('/get-leaderboard', { session_id: sessionId });
  }

  async getSessionLogs(sessionId: string): Promise<LogsResponse> {
    switch (this.agentType) {
      case 'loading':
        const loadingResponse = await this.request<{success: boolean, data?: any, error?: string}>('/get-internal-messages', { session_id: sessionId });
        return { 
          logs: loadingResponse.success && loadingResponse.data ? [JSON.stringify(loadingResponse.data)] : [], 
          messages: loadingResponse.success && loadingResponse.data ? [JSON.stringify(loadingResponse.data)] : []
        };
      case 'cleaning':
        const response = await this.request<{success: boolean, data?: string, error?: string}>('/get-logs', { session_id: sessionId });
        return { 
          logs: response.success && response.data ? [response.data] : [], 
          messages: response.success && response.data ? [response.data] : []
        };
      default:
        return { logs: [], messages: [] };
    }
  }

  async getSessionRecommendations(sessionId: string): Promise<RecommendationsResponse> {
    switch (this.agentType) {
      case 'cleaning':
        const response = await this.request<{success: boolean, data?: string, error?: string}>('/get-cleaning-steps', { session_id: sessionId });
        return { 
          recommendations: response.success && response.data ? [response.data] : [],
          cleaning_steps: response.success && response.data ? [response.data] : []
        };
      case 'visualization':
        const vizResponse = await this.request<{success: boolean, data?: string, error?: string}>('/get-visualization-steps', { session_id: sessionId });
        return { 
          recommendations: vizResponse.success && vizResponse.data ? [vizResponse.data] : [],
          visualization_steps: vizResponse.success && vizResponse.data ? [vizResponse.data] : []
        };
      case 'engineering':
        const engResponse = await this.request<{success: boolean, data?: string, error?: string}>('/get-engineering-steps', { session_id: sessionId });
        return { 
          recommendations: engResponse.success && engResponse.data ? [engResponse.data] : [],
          engineering_steps: engResponse.success && engResponse.data ? [engResponse.data] : []
        };
      case 'training':
        const trainResponse = await this.request<{success: boolean, data?: string, error?: string}>('/get-ml-steps', { session_id: sessionId });
        return { 
          recommendations: trainResponse.success && trainResponse.data ? [trainResponse.data] : [],
          ml_steps: trainResponse.success && trainResponse.data ? [trainResponse.data] : []
        };
      default:
        return { recommendations: [] };
    }
  }

  async getSessionAnalysis(sessionId: string): Promise<AnalysisResponse> {
    if (this.agentType === 'prediction') {
      return this.request<AnalysisResponse>('/get-model-analysis', { session_id: sessionId });
    }
    return { analysis: '' };
  }

  // Visualization specific methods

  // Data Loader specific methods
  async getSessionAIMessage(sessionId: string): Promise<{success: boolean, data?: string, error?: string}> {
    if (this.agentType !== 'loading') {
      throw new Error('AI message only available for loading agent');
    }
    return this.request<{success: boolean, data?: string, error?: string}>('/get-ai-message', { session_id: sessionId });
  }

  async getSessionToolCalls(sessionId: string): Promise<{success: boolean, data?: any, error?: string}> {
    if (this.agentType !== 'loading') {
      throw new Error('Tool calls only available for loading agent');
    }
    return this.request<{success: boolean, data?: any, error?: string}>('/get-tool-calls', { session_id: sessionId });
  }

  async getSessionFullResponse(sessionId: string): Promise<{success: boolean, data?: any, error?: string}> {
    if (this.agentType !== 'loading') {
      throw new Error('Full response only available for loading agent');
    }
    return this.request<{success: boolean, data?: any, error?: string}>('/get-full-response', { session_id: sessionId });
  }

  // Cleanup data handler for Data Cleaning agent (special POST endpoint)
  async getCleanedData(sessionId: string): Promise<DataResponse> {
    if (this.agentType !== 'cleaning') {
      throw new Error('Cleaned data only available for cleaning agent');
    }
    return this.request<DataResponse>('/get-cleaned-data', { session_id: sessionId });
  }
}

// Singleton clients for each agent type
export const dataLoaderClient = new UAgentClient('loading');
export const dataCleaningClient = new UAgentClient('cleaning');  
export const visualizationClient = new UAgentClient('visualization');
export const featureEngineeringClient = new UAgentClient('engineering');
export const trainingClient = new UAgentClient('training');
export const predictionClient = new UAgentClient('prediction');

// Helper function to get client by agent type
export function getAgentClient(agentType: AgentType): UAgentClient {
  switch (agentType) {
    case 'loading':
      return dataLoaderClient;
    case 'cleaning':
      return dataCleaningClient;
    case 'visualization':
      return visualizationClient;  
    case 'engineering':
      return featureEngineeringClient;
    case 'training':
      return trainingClient;
    case 'prediction':
      return predictionClient;
    default:
      throw new Error(`Unknown agent type: ${agentType}`);
  }
}
