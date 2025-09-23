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

import type { SessionMeta } from './store';

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

export interface EngineerFeaturesParams {
  session_id?: string;
  filename?: string;
  file_content?: string; // Base64 encoded CSV
  target_variable: string;
  user_instructions?: string;
  feature_options?: {
    create_polynomial?: boolean;
    create_interactions?: boolean;
    normalize_features?: boolean;
    handle_categorical?: 'onehot' | 'label' | 'target';
    create_datetime_features?: boolean;
  };
}

export interface TrainingParams {
  session_id?: string;
  filename?: string;
  file_content?: string; // Base64 encoded CSV
  target_column: string;
  time_budget_seconds?: number;
  user_instructions?: string;
}

export interface TrainModelParams {
  session_id?: string;
  filename?: string;
  file_content?: string; // Base64 encoded CSV
  target_variable: string;
  user_instructions?: string;
  max_runtime_secs?: number;
  cv_folds?: number;
  balance_classes?: boolean;
  max_models?: number;
  exclude_algos?: string[];
  seed?: number;
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

export interface PredictSingleParams {
  model_session_id?: string;
  model_path?: string;
  input_data: Record<string, unknown>;
}

export interface PredictBatchParams {
  model_session_id?: string;
  model_path?: string;
  filename?: string;
  file_content?: string; // Base64 encoded CSV
}

export interface AnalyzeModelParams {
  model_session_id?: string;
  model_path?: string;
  query: string;
}

// Result response types  
export interface DataResponse {
  data?: {
    records: Array<Record<string, unknown>>;
    columns: string[];
  };
  success?: boolean;
  error?: string;
}

export interface CodeResponse {
  code?: string;
  generated_code?: string; // Backend uses this field name
}

export interface ChartResponse {
  figure?: unknown; // Plotly JSON
  plotly_chart?: unknown; // Plotly JSON (alternative property name)
  chart_type?: string;
  success?: boolean;
  error?: string;
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

// Configuration for agent URLs - can be overridden via environment
const DEFAULT_HOST = typeof window !== 'undefined' 
  ? (window.location.hostname === 'localhost' ? '127.0.0.1' : window.location.hostname)
  : '127.0.0.1'; // Use current hostname in browser, localhost in server

// Configuration logged only in browser console (avoiding hydration issues)
// Check console for uAgent client URLs if debugging is needed
const AGENT_PORTS: Record<AgentType, number> = {
  loading: 8005,
  cleaning: 8004, 
  visualization: 8006,
  engineering: 8007,
  training: 8008,
  prediction: 8009,
};

// Base URLs for each agent with environment override support
const AGENT_BASE_URLS: Record<AgentType, string> = {
  loading: `http://${DEFAULT_HOST}:${AGENT_PORTS.loading}`,
  cleaning: `http://${DEFAULT_HOST}:${AGENT_PORTS.cleaning}`, 
  visualization: `http://${DEFAULT_HOST}:${AGENT_PORTS.visualization}`,
  engineering: `http://${DEFAULT_HOST}:${AGENT_PORTS.engineering}`,
  training: '/api/training', // Use backend proxy for training
  prediction: `http://${DEFAULT_HOST}:${AGENT_PORTS.prediction}`,
};

// Main backend API URL
const BACKEND_API_URL = `http://${DEFAULT_HOST}:8000/api`;

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
    const url = `${this.baseUrl}${endpoint}`;

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: data ? JSON.stringify(data) : undefined,
      });

      if (!response.ok) {
        const errorText = await response.text();
        const error = new Error(`Request failed: ${response.status} ${response.statusText}: ${errorText}`);
        throw error;
      }

      return response.json();
    } catch (error) {
      throw error;
    }
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
      return this.request<SessionResponse>('/clean-from-session', params);
    } else {
      return this.request<SessionResponse>('/clean-csv', params);
    }
  }

  // Visualization operations (8006)  
  async createChartDirect(params: VizParams): Promise<ChartResponse> {
    return this.request<ChartResponse>('/create-chart-direct', {
      filename: params.filename,
      file_content: params.file_content,
      user_instructions: params.user_instructions,
      max_retries: 3
    });
  }

  async createChart(params: VizParams): Promise<SessionResponse> {
    if (params.session_id) {
      return this.request<SessionResponse>('/create-chart', params);
    } else {
      return this.request<SessionResponse>('/create-chart-csv', params);
    }
  }

  // Feature Engineering operations (8007)
  async engineerFeatures(params: EngineerFeaturesParams): Promise<SessionResponse> {
    if (params.session_id) {
      return this.request<SessionResponse>('/engineer-features', params);
    } else {
      return this.request<SessionResponse>('/engineer-features-csv', params);
    }
  }

  // ML Training operations (8008)
  async trainModel(params: TrainModelParams): Promise<SessionResponse> {
    if (params.session_id) {
      return this.request<SessionResponse>('/train-model-from-session', {
        source_session_id: params.session_id,
        target_variable: params.target_variable,
        user_instructions: params.user_instructions,
        max_runtime_secs: params.max_runtime_secs,
        cv_folds: params.cv_folds,
        balance_classes: params.balance_classes,
        max_models: params.max_models,
        exclude_algos: params.exclude_algos,
        seed: params.seed,
      });
    } else {
      return this.request<SessionResponse>('/train-model-csv', params);
    }
  }

  // ML Prediction operations (8009)
  async predictSingle(params: PredictSingleParams): Promise<SessionResponse> {
    return this.request<SessionResponse>('/predict-single', params);
  }

  async predictBatch(params: PredictBatchParams): Promise<SessionResponse> {
    return this.request<SessionResponse>('/predict-batch', params);
  }

  async analyzeModel(params: AnalyzeModelParams): Promise<SessionResponse> {
    return this.request<SessionResponse>('/analyze-model', params);
  }

  // Session result getters  
  async getSessionData(sessionId: string): Promise<DataResponse> {
    switch (this.agentType) {
      case 'loading':
        return this.request<DataResponse>('/get-artifacts', { session_id: sessionId });
      case 'cleaning':
        return this.request<DataResponse>('/get-cleaned-data', { session_id: sessionId });
      case 'engineering':
        return this.request<DataResponse>('/get-session-data', { session_id: sessionId });
      case 'training':
        return this.request<DataResponse>('/get-original-data', { session_id: sessionId });
      case 'prediction':
        return this.request<DataResponse>('/get-prediction-results', { session_id: sessionId });
      default:
        return this.getSessionResult<DataResponse>(`/session/${sessionId}/data`);
    }
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
    const response = await this.request<{success: boolean, plotly_chart?: any, chart_type?: string, error?: string}>('/get-plotly-graph', { session_id: sessionId });
    return { 
      figure: response.plotly_chart // Map plotly_chart to figure for frontend
    };
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
      case 'engineering':
        const engResponse = await this.request<{success: boolean, data?: string, error?: string}>('/get-logs', { session_id: sessionId });
        return { 
          logs: engResponse.success && engResponse.data ? [engResponse.data] : [], 
          messages: engResponse.success && engResponse.data ? [engResponse.data] : []
        };
      case 'training':
        const trainLogResponse = await this.request<{success: boolean, data?: string, error?: string}>('/get-logs', { session_id: sessionId });
        return { 
          logs: trainLogResponse.success && trainLogResponse.data ? [trainLogResponse.data] : [], 
          messages: trainLogResponse.success && trainLogResponse.data ? [trainLogResponse.data] : []
        };
      case 'prediction':
        const predLogResponse = await this.request<{success: boolean, data?: string, error?: string}>('/get-logs', { session_id: sessionId });
        return { 
          logs: predLogResponse.success && predLogResponse.data ? [predLogResponse.data] : [], 
          messages: predLogResponse.success && predLogResponse.data ? [predLogResponse.data] : []
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
      case 'prediction':
        const predResponse = await this.request<{success: boolean, data?: string, error?: string}>('/get-model-analysis', { session_id: sessionId });
        return { 
          recommendations: predResponse.success && predResponse.data ? [predResponse.data] : []
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

  // 🔥 ML TRAINING AGENT SPECIFIC METHODS 🔥
  async getLeaderboard(sessionId: string): Promise<LeaderboardResponse> {
    if (this.agentType !== 'training') {
      throw new Error('Leaderboard only available for training agent');
    }
    return this.request<LeaderboardResponse>('/get-leaderboard', { session_id: sessionId });
  }

  async getBestModelId(sessionId: string): Promise<{success: boolean, model_id?: string, error?: string}> {
    if (this.agentType !== 'training') {
      throw new Error('Best model ID only available for training agent');
    }
    return this.request<{success: boolean, model_id?: string, error?: string}>('/get-best-model-id', { session_id: sessionId });
  }

  async getModelPath(sessionId: string): Promise<{success: boolean, model_path?: string, error?: string}> {
    if (this.agentType !== 'training') {
      throw new Error('Model path only available for training agent');
    }
    return this.request<{success: boolean, model_path?: string, error?: string}>('/get-model-path', { session_id: sessionId });
  }

  async getTrainingFunction(sessionId: string): Promise<CodeResponse> {
    if (this.agentType !== 'training') {
      throw new Error('Training function only available for training agent');
    }
    return this.request<CodeResponse>('/get-training-function', { session_id: sessionId });
  }

  async getWorkflowSummary(sessionId: string): Promise<{success: boolean, data?: any, error?: string}> {
    if (this.agentType !== 'training') {
      throw new Error('Workflow summary only available for training agent');
    }
    return this.request<{success: boolean, data?: any, error?: string}>('/get-workflow-summary', { session_id: sessionId });
  }

  async getTrainingFullResponse(sessionId: string): Promise<{success: boolean, data?: any, error?: string}> {
    if (this.agentType !== 'training') {
      throw new Error('Full response only available for training agent');
    }
    return this.request<{success: boolean, data?: any, error?: string}>('/get-training-full-response', { session_id: sessionId });
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

// Session synchronization functions
export async function getBackendSessions(agentType?: AgentType): Promise<SessionMeta[]> {
  try {
    const url = agentType 
      ? `${BACKEND_API_URL}/agents/sessions?agent_type=${agentType}`
      : `${BACKEND_API_URL}/agents/sessions`;
      
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch sessions: ${response.statusText}`);
    }
    
    const sessions = await response.json();
    
    // Convert backend session format to frontend SessionMeta format
    return sessions.map((session: any) => ({
      sessionId: session.session_id,
      agentType: session.agent_type as AgentType,
      createdAt: new Date(session.created_at * 1000).toISOString(), // Convert Unix timestamp
      status: session.status as 'completed',
      description: session.description,
    }));
  } catch (error) {
    console.error('Failed to fetch backend sessions:', error);
    return [];
  }
}

export async function syncSessionsWithBackend(): Promise<SessionMeta[]> {
  try {
    return await getBackendSessions();
  } catch (error) {
    console.error('Session sync failed:', error);
    return [];
  }
}
