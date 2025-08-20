import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { AgentType, SessionId } from './uagent-client'

// Session metadata interface
export interface SessionMeta {
  sessionId: SessionId;
  agentType: AgentType;
  createdAt: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  labels?: string[];
  description?: string;
  executionTimeSeconds?: number;
  error?: string;
}

// Workflow step interface
export interface WorkflowStep {
  id: string;
  agentType: AgentType;
  parameters?: Record<string, unknown>;
  sessionId?: SessionId;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  error?: string;
}

// Workflow chain interface
export interface WorkflowChain {
  id: string;
  name: string;
  steps: WorkflowStep[];
  currentStepIndex: number;
  status: 'draft' | 'running' | 'paused' | 'completed' | 'failed';
  createdAt: string;
}

// Upload queue item
export interface UploadItem {
  id: string;
  file: File;
  status: 'queued' | 'uploading' | 'completed' | 'failed';
  progress?: number;
  error?: string;
}

// UI state
export interface UIState {
  activeAgent: AgentType | null;
  currentTab: string;
  sidebarOpen: boolean;
  uploadModalOpen: boolean;
  workflowBuilderOpen: boolean;
}

// Store interface
interface AppStore {
  // Session management
  sessions: SessionMeta[];
  addSession: (session: SessionMeta) => void;
  updateSession: (sessionId: SessionId, updates: Partial<SessionMeta>) => void;
  removeSession: (sessionId: SessionId) => void;
  getSession: (sessionId: SessionId) => SessionMeta | undefined;
  getSessionsByAgent: (agentType: AgentType) => SessionMeta[];
  clearSessions: () => void;
  
  // Workflow chains
  workflows: WorkflowChain[];
  activeWorkflow: WorkflowChain | null;
  createWorkflow: (name: string, steps: Omit<WorkflowStep, 'id' | 'status'>[]) => WorkflowChain;
  updateWorkflow: (workflowId: string, updates: Partial<WorkflowChain>) => void;
  updateWorkflowStep: (workflowId: string, stepId: string, updates: Partial<WorkflowStep>) => void;
  setActiveWorkflow: (workflow: WorkflowChain | null) => void;
  removeWorkflow: (workflowId: string) => void;
  
  // Upload queue
  uploadQueue: UploadItem[];
  addToUploadQueue: (file: File) => string;
  updateUploadItem: (id: string, updates: Partial<UploadItem>) => void;
  removeFromUploadQueue: (id: string) => void;
  clearUploadQueue: () => void;
  
  // UI state
  ui: UIState;
  setActiveAgent: (agentType: AgentType | null) => void;
  setCurrentTab: (tab: string) => void;
  setSidebarOpen: (open: boolean) => void;
  setUploadModalOpen: (open: boolean) => void;
  setWorkflowBuilderOpen: (open: boolean) => void;
}

// Generate unique IDs
const generateId = () => Math.random().toString(36).substring(2) + Date.now().toString(36);

export const useAppStore = create<AppStore>()(
  persist(
    (set, get) => ({
      // Initial state
      sessions: [],
      workflows: [],
      activeWorkflow: null,
      uploadQueue: [],
      ui: {
        activeAgent: null,
        currentTab: 'data',
        sidebarOpen: true,
        uploadModalOpen: false,
        workflowBuilderOpen: false,
      },
      
      // Session management actions
      addSession: (session) => {
        set((state) => ({
          sessions: [session, ...state.sessions].slice(0, 100), // Keep latest 100 sessions
        }))
      },
      
      updateSession: (sessionId, updates) => {
        set((state) => ({
          sessions: state.sessions.map((session) =>
            session.sessionId === sessionId ? { ...session, ...updates } : session
          ),
        }))
      },
      
      removeSession: (sessionId) => {
        set((state) => ({
          sessions: state.sessions.filter((session) => session.sessionId !== sessionId),
        }))
      },
      
      getSession: (sessionId) => {
        return get().sessions.find((session) => session.sessionId === sessionId)
      },
      
      getSessionsByAgent: (agentType) => {
        return get().sessions.filter((session) => session.agentType === agentType)
      },
      
      clearSessions: () => {
        set({ sessions: [] })
      },
      
      // Workflow chain actions
      createWorkflow: (name, steps) => {
        const workflow: WorkflowChain = {
          id: generateId(),
          name,
          steps: steps.map((step) => ({
            ...step,
            id: generateId(),
            status: 'pending',
          })),
          currentStepIndex: 0,
          status: 'draft',
          createdAt: new Date().toISOString(),
        }
        
        set((state) => ({
          workflows: [workflow, ...state.workflows],
          activeWorkflow: workflow,
        }))
        
        return workflow
      },
      
      updateWorkflow: (workflowId, updates) => {
        set((state) => ({
          workflows: state.workflows.map((workflow) =>
            workflow.id === workflowId ? { ...workflow, ...updates } : workflow
          ),
          activeWorkflow:
            state.activeWorkflow?.id === workflowId
              ? { ...state.activeWorkflow, ...updates }
              : state.activeWorkflow,
        }))
      },
      
      updateWorkflowStep: (workflowId, stepId, updates) => {
        set((state) => ({
          workflows: state.workflows.map((workflow) =>
            workflow.id === workflowId
              ? {
                  ...workflow,
                  steps: workflow.steps.map((step) =>
                    step.id === stepId ? { ...step, ...updates } : step
                  ),
                }
              : workflow
          ),
          activeWorkflow:
            state.activeWorkflow?.id === workflowId
              ? {
                  ...state.activeWorkflow,
                  steps: state.activeWorkflow.steps.map((step) =>
                    step.id === stepId ? { ...step, ...updates } : step
                  ),
                }
              : state.activeWorkflow,
        }))
      },
      
      setActiveWorkflow: (workflow) => {
        set({ activeWorkflow: workflow })
      },
      
      removeWorkflow: (workflowId) => {
        set((state) => ({
          workflows: state.workflows.filter((workflow) => workflow.id !== workflowId),
          activeWorkflow:
            state.activeWorkflow?.id === workflowId ? null : state.activeWorkflow,
        }))
      },
      
      // Upload queue actions
      addToUploadQueue: (file) => {
        const id = generateId()
        const uploadItem: UploadItem = {
          id,
          file,
          status: 'queued',
          progress: 0,
        }
        
        set((state) => ({
          uploadQueue: [...state.uploadQueue, uploadItem],
        }))
        
        return id
      },
      
      updateUploadItem: (id, updates) => {
        set((state) => ({
          uploadQueue: state.uploadQueue.map((item) =>
            item.id === id ? { ...item, ...updates } : item
          ),
        }))
      },
      
      removeFromUploadQueue: (id) => {
        set((state) => ({
          uploadQueue: state.uploadQueue.filter((item) => item.id !== id),
        }))
      },
      
      clearUploadQueue: () => {
        set({ uploadQueue: [] })
      },
      
      // UI state actions
      setActiveAgent: (agentType) => {
        set((state) => ({
          ui: { ...state.ui, activeAgent: agentType },
        }))
      },
      
      setCurrentTab: (tab) => {
        set((state) => ({
          ui: { ...state.ui, currentTab: tab },
        }))
      },
      
      setSidebarOpen: (open) => {
        set((state) => ({
          ui: { ...state.ui, sidebarOpen: open },
        }))
      },
      
      setUploadModalOpen: (open) => {
        set((state) => ({
          ui: { ...state.ui, uploadModalOpen: open },
        }))
      },
      
      setWorkflowBuilderOpen: (open) => {
        set((state) => ({
          ui: { ...state.ui, workflowBuilderOpen: open },
        }))
      },
    }),
    {
      name: 'ai-data-science-platform',
      // Only persist sessions and workflows, not UI state
      partialize: (state) => ({
        sessions: state.sessions,
        workflows: state.workflows,
      }),
    }
  )
)

// Selector hooks for better performance
export const useSessionsStore = () => useAppStore((state) => ({
  sessions: state.sessions,
  addSession: state.addSession,
  updateSession: state.updateSession,
  removeSession: state.removeSession,
  getSession: state.getSession,
  getSessionsByAgent: state.getSessionsByAgent,
  clearSessions: state.clearSessions,
}))

export const useWorkflowStore = () => useAppStore((state) => ({
  workflows: state.workflows,
  activeWorkflow: state.activeWorkflow,
  createWorkflow: state.createWorkflow,
  updateWorkflow: state.updateWorkflow,
  updateWorkflowStep: state.updateWorkflowStep,
  setActiveWorkflow: state.setActiveWorkflow,
  removeWorkflow: state.removeWorkflow,
}))

export const useUploadStore = () => useAppStore((state) => ({
  uploadQueue: state.uploadQueue,
  addToUploadQueue: state.addToUploadQueue,
  updateUploadItem: state.updateUploadItem,
  removeFromUploadQueue: state.removeFromUploadQueue,
  clearUploadQueue: state.clearUploadQueue,
}))

export const useUIStore = () => useAppStore((state) => ({
  ui: state.ui,
  setActiveAgent: state.setActiveAgent,
  setCurrentTab: state.setCurrentTab,
  setSidebarOpen: state.setSidebarOpen,
  setUploadModalOpen: state.setUploadModalOpen,
  setWorkflowBuilderOpen: state.setWorkflowBuilderOpen,
}))
