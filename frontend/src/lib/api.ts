import { ChatMessage } from '@/SimContext';
import { backendUrl } from './utils';
import axios from 'axios';

export const api = axios.create({
    baseURL: `${backendUrl}`,
});


export namespace apis {

    export interface EventConfig {
        name: string;
        policy: string;
        websearch: string;
        description: string;
    }

    export interface AgentConfig {
        firstName: string;
        lastName: string;
        age: number;
        avatar: string;
        dailyPlan: string;
        innate: string;
        learned: string;
    }

    export interface Meta {
        template_sim_code: string;
        name: string;
        bullets: string[];
        description: string;
        start_date: string;
        curr_time: string;
        sec_per_step: number;
        maze_name: string;
        persona_names: string[];
        sim_mode: string,
        step: number;
    }


    export interface Agent {
        curr_time?: number;
        curr_tile?: string;
        daily_plan_req: string;
        name: string;
        first_name: string;
        last_name: string;
        age: number;
        innate: string;
        learned: string;
        currently: string;
        lifestyle: string;
        living_area: string;
        daily_req?: string[];
        f_daily_schedule?: string[];
        f_daily_schedule_hourly_org?: string[];
        act_address?: string;
        act_start_time?: string;
        act_duration?: string;
        act_description?: string;
        act_pronunciatio?: string;
        act_event?: [string, string?, string?];
        act_obj_description?: string;
        act_obj_pronunciatio?: string;
        act_obj_event?: [string?, string?, string?];
        chatting_with?: string;
        chat?: string[][];
        chatting_with_buffer?: Record<string, string>;
        chatting_end_time?: string;
        act_path_set?: boolean;
        planned_path?: string[];
        avatar?: string;
        plan?: string[];
        memory?: string[];
        bibliography?: string;

        // New fields from Scratch class
        vision_r: number;
        att_bandwidth: number;
        retention: number;
        concept_forget: number;
        daily_reflection_time: number;
        daily_reflection_size: number;
        overlap_reflect_th: number;
        kw_strg_event_reflect_th: number;
        kw_strg_thought_reflect_th: number;
        recency_w: number;
        relevance_w: number;
        importance_w: number;
        recency_decay: number;
        importance_trigger_max: number;
        importance_trigger_curr: number;
        importance_ele_n: number;
        thought_count: number;
    }

    export interface LLMConfig {
        type: string;
        baseUrl: string;
        key: string;
        engine: string;
        temperature: number;
        maxTokens: number;
        topP: number;
        freqPenalty: number;
        presPenalty: number;
        stream: boolean;
    }

    export interface Event {
        name: string;
        policy: string;
        websearch: string;
        description: string;
    }


    export interface Template {
        simCode: string;
        events: Event[];
        personas: Agent[];
        meta: Meta;
    }

    export interface TemplateListItem {
        template_sim_code: string;
        name: string;
        bullets: string[];
        description: string;
        start_date: string;
        curr_time: string;
        sec_per_step: number;
        maze_name: string;
        persona_names: string[];
        step: number;
        sim_mode: string;
    }

    function isEmptyObject(obj: Record<string, any> | any[]): boolean {
        if (Array.isArray(obj)) {
            return false;
        }
        return Object.keys(obj).length === 0;
    }

    export const fetchTemplate = async (templateName: string): Promise<Template> => {
        try {
            const response = await api.get<{ meta: any, events: any[], personas: Record<string, any> }>('/fetch_template', { params: { sim_code: templateName } });
            const { meta, events, personas } = response.data;
            return {
                simCode: templateName,
                events: isEmptyObject(events) ? [] : events.map(event => ({
                    name: event.name,
                    policy: event.policy,
                    websearch: event.websearch,
                    description: event.description,
                })),
                personas: Object.values(personas).map(persona => ({
                    curr_time: undefined,
                    curr_tile: undefined,
                    daily_plan_req: persona.daily_plan_req,
                    name: persona.name,
                    first_name: persona.first_name,
                    last_name: persona.last_name,
                    age: persona.age,
                    innate: persona.innate,
                    learned: persona.learned,
                    currently: persona.currently,
                    lifestyle: persona.lifestyle,
                    living_area: persona.living_area,
                    daily_req: [],
                    f_daily_schedule: [],
                    f_daily_schedule_hourly_org: [],
                    act_address: undefined,
                    act_start_time: undefined,
                    act_duration: undefined,
                    act_description: undefined,
                    act_pronunciatio: undefined,
                    act_event: [persona.name, undefined, undefined],
                    act_obj_description: undefined,
                    act_obj_pronunciatio: undefined,
                    act_obj_event: [undefined, undefined, undefined],
                    chatting_with: undefined,
                    chat: [[]],
                    chatting_with_buffer: {},
                    chatting_end_time: undefined,
                    act_path_set: false,
                    planned_path: [],

                    // New fields from Scratch class
                    vision_r: 4,
                    att_bandwidth: 3,
                    retention: 5,
                    concept_forget: 100,
                    daily_reflection_time: 60 * 3,
                    daily_reflection_size: 5,
                    overlap_reflect_th: 2,
                    kw_strg_event_reflect_th: 4,
                    kw_strg_thought_reflect_th: 4,
                    recency_w: 1,
                    relevance_w: 1,
                    importance_w: 1,
                    recency_decay: 0.99,
                    importance_trigger_max: 150,
                    importance_trigger_curr: 150, // Using importance_trigger_max as initial value
                    importance_ele_n: 0,
                    thought_count: 5,
                })),
                meta,
            };
        } catch (error) {
            console.error("Error fetching template:", error);
            throw error;
        }
    };

    export const fetchTemplates = async (): Promise<{ envs: TemplateListItem[], all_templates: string[] }> => {
        console.log(backendUrl)
        try {
            const response = await api.get<{ envs: TemplateListItem[], all_templates: string[] }>('/fetch_templates');

            // 定义优先级顺序
            const priorityOrder = ['shbz', 'legislative_council', 'dragon_tv_demo'];

            // 对 envs 数组排序（按 template_sim_code 的优先级）
            response.data.envs.sort((a, b) => {
                const aCode = a.template_sim_code;
                const bCode = b.template_sim_code;

                const aPriority = priorityOrder.indexOf(aCode);
                const bPriority = priorityOrder.indexOf(bCode);

                if (aPriority !== -1 && bPriority !== -1) {
                    return aPriority - bPriority; // 优先级高的排在前面
                }
                if (aPriority !== -1) return -1; // a 在优先级列表中，排在前面
                if (bPriority !== -1) return 1; // b 在优先级列表中，a 排在后面

                // 都不在优先级列表中，按字典序排序
                return aCode.localeCompare(bCode);
            });

            // 对 all_templates 数组排序（按字符串的优先级）
            response.data.all_templates.sort((a, b) => {
                const aPriority = priorityOrder.indexOf(a);
                const bPriority = priorityOrder.indexOf(b);

                if (aPriority !== -1 && bPriority !== -1) {
                    return aPriority - bPriority;
                }
                if (aPriority !== -1) return -1;
                if (bPriority !== -1) return 1;

                // 都不在优先级列表中，按字典序排序
                return a.localeCompare(b);
            });

            return response.data;
        } catch (error) {
            console.error("Error fetching templates:", error);
            throw error;
        }
    };


    export const startSim = async (
        simCode: string,
        template: apis.Template,
        llmConfig: apis.LLMConfig,
        initialRounds: number
    ): Promise<any> => {
        try {
            const response = await api.post('/start', {
                simCode,
                template,
                llmConfig,
                initialRounds
            });
            return response.data;
        } catch (error) {
            console.error("Error starting simulation:", error);
            throw error;
        }
    };

    export const runSim = async (count: number, simCode: string): Promise<any> => {
        try {
            const response = await api.get(`/run`, { params: { count, sim_code: simCode } });
            return response.data;
        } catch (error) {
            console.error("Error running simulation:", error);
            throw error;
        }
    };

    export const updateEnv = async (updateData: any, simCode: string): Promise<any> => {
        try {
            const response = await api.post(`/update_env`, updateData, { params: { sim_code: simCode } });
            return response.data;
        } catch (error) {
            console.error("Error updating environment:", error);
            throw error;
        }
    };

    export const agentsInfo = async (simCode: string): Promise<Agent[]> => {
        try {
            const response = await api.get(`/personas_info`, { params: { sim_code: simCode } });
            return response.data.personas;
        } catch (error) {
            console.error("Error fetching agents info:", error);
            throw error;
        }
    };

    export const agentDetail = async (simCode: string, agentName: string): Promise<{
        scratch: Agent,
        a_mem: Record<string, string>,
        s_mem: Record<string, string>,
    }> => {
        try {
            const response = await api.get(`/persona_detail`, { params: { sim_code: simCode, agent_name: agentName } });
            return response.data;
        } catch (error) {
            console.error("Error fetching agent detail:", error);
            throw error;
        }
    };

    export const sendCommand = async (command: string, simCode: string): Promise<any> => {
        try {
            const response = await api.get(`/command`, { params: { command, sim_code: simCode } });
            return response.data;
        } catch (error) {
            console.error("Error sending command:", error);
            throw error;
        }
    };

    export const privateChat = async (
        simCode: string,
        person: string,
        type: 'interview' | 'whisper',
        history: ChatMessage[],
        content: string
    ): Promise<any> => {
        try {
            const formattedHistory: [string, string][] = history.map(msg => [
                msg.role === 'agent' ? person : 'Interviewer',
                msg.content
            ]);

            const response = await api.post(`/chat`, {
                agent_name: person,
                type,
                history: formattedHistory,
                content
            }, { params: { sim_code: simCode } });
            return response.data;
        } catch (error) {
            console.error("Error sending private chat:", error);
            throw error;
        }
    }

    export const publishEvent = async (eventData: EventConfig, simCode: string): Promise<any> => {
        try {
            const response = await api.post(`/publish_events`, eventData, { params: { sim_code: simCode } });
            return response.data;
        } catch (error) {
            console.error("Error publishing event:", error);
            throw error;
        }
    };

    export const queryStatus = async (simCode: string): Promise<'running' | 'stopped' | 'started'> => {
        try {
            const response = await api.get(`/status`, { params: { sim_code: simCode } });
            return response.data.status;
        } catch (error) {
            console.error("Error querying status:", error);
            throw error;
        }
    }

    export const messageSocket = (simCode: string) => {
        return new WebSocket(`${backendUrl}/ws?sim_code=${simCode}`);
    }


}

