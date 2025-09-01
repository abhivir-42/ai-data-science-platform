/**
 * Workflow Client for executing multi-agent workflows
 * 
 * This client handles communication with the backend workflow execution API
 * allowing the frontend to orchestrate complex multi-step data science workflows.
 */

// Base response types
export interface WorkflowExecutionResponse {
  success: boolean;
  workflow_id: string;
  message: string;
  status: string;
}

export interface WorkflowStatusResponse {
  id: string;
  name: string;
  status: string;
  current_step: number;
  total_steps: number;
  progress_percentage: number;
  execution_time: number;
  steps: Array<{
    agent_type: string;
    status: string;
    session_id?: string;
    error?: string;
  }>;
}

export interface WorkflowResultsResponse {
  workflow_id: string;
  name: string;
  status: string;
  total_execution_time: number;
  results: any;
  steps: Array<{
    id: string;
    agent_type: string;
    status: string;
    session_id?: string;
    execution_time_seconds?: number;
    error?: string;
  }>;
}

export interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  steps: string[];
  estimated_time: string;
  difficulty: string;
}

export class WorkflowClient {
  private baseUrl: string;
  
  constructor(baseUrl: string = 'http://localhost:8000/api') {
    this.baseUrl = baseUrl;
  }

  // Helper for making requests
  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Request failed: ${response.status} ${response.statusText}: ${errorText}`);
    }

    return response.json();
  }

  private async uploadRequest<T>(endpoint: string, formData: FormData): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Upload request failed: ${response.status} ${response.statusText}: ${errorText}`);
    }

    return response.json();
  }

  // Execute Quick Data Analysis workflow with file upload
  async executeQuickAnalysis(
    file: File, 
    userInstructions: string = 'Perform quick data analysis on the uploaded file'
  ): Promise<WorkflowExecutionResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_instructions', userInstructions);
    
    return this.uploadRequest<WorkflowExecutionResponse>('/workflows/execute-quick-analysis', formData);
  }

  // Execute Complete ML Pipeline workflow with file upload
  async executeMLPipeline(
    file: File,
    targetVariable: string,
    userInstructions: string = 'Train machine learning models on the uploaded dataset',
    maxRuntimeSecs: number = 300
  ): Promise<WorkflowExecutionResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_variable', targetVariable);
    formData.append('user_instructions', userInstructions);
    formData.append('max_runtime_secs', maxRuntimeSecs.toString());
    
    return this.uploadRequest<WorkflowExecutionResponse>('/workflows/execute-ml-pipeline', formData);
  }

  // Execute custom workflow
  async executeWorkflow(
    name: string,
    steps: Array<{
      agent_type: string;
      parameters: Record<string, any>;
    }>,
    initialData?: Record<string, any>
  ): Promise<WorkflowExecutionResponse> {
    return this.request<WorkflowExecutionResponse>('/workflows/execute', {
      method: 'POST',
      body: JSON.stringify({
        name,
        steps,
        initial_data: initialData,
      }),
    });
  }

  // Get workflow execution status
  async getWorkflowStatus(workflowId: string): Promise<WorkflowStatusResponse> {
    return this.request<WorkflowStatusResponse>(`/workflows/${workflowId}/status`);
  }

  // Get workflow execution results
  async getWorkflowResults(workflowId: string): Promise<WorkflowResultsResponse> {
    return this.request<WorkflowResultsResponse>(`/workflows/${workflowId}/results`);
  }

  // List all workflow executions
  async listWorkflows(): Promise<Array<{
    id: string;
    name: string;
    status: string;
    created_at: number;
    total_execution_time?: number;
    total_steps: number;
    completed_steps: number;
  }>> {
    return this.request<any[]>('/workflows/');
  }

  // Get workflow templates
  async getWorkflowTemplates(): Promise<WorkflowTemplate[]> {
    return this.request<WorkflowTemplate[]>('/workflows/templates');
  }

  // Check health of workflow agents
  async checkAgentsHealth(): Promise<{
    all_agents_healthy: boolean;
    agent_health: Record<string, any>;
    workflow_service_status: string;
  }> {
    return this.request<any>('/workflows/health');
  }

  // Poll workflow status until completion
  async pollWorkflowUntilComplete(
    workflowId: string,
    onProgress?: (status: WorkflowStatusResponse) => void,
    pollIntervalMs: number = 2000,
    timeoutMs: number = 600000 // 10 minutes
  ): Promise<WorkflowResultsResponse> {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeoutMs) {
      const status = await this.getWorkflowStatus(workflowId);
      
      if (onProgress) {
        onProgress(status);
      }
      
      if (status.status === 'completed') {
        return this.getWorkflowResults(workflowId);
      }
      
      if (status.status === 'failed') {
        throw new Error(`Workflow failed: ${status.steps.find(s => s.error)?.error || 'Unknown error'}`);
      }
      
      // Wait before polling again
      await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
    }
    
    throw new Error(`Workflow polling timed out after ${timeoutMs}ms`);
  }
}

// Default workflow client instance
export const workflowClient = new WorkflowClient();
