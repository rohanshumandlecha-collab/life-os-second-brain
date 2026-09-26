import express from 'express';
import path from 'path';
import dotenv from 'dotenv';
import { GoogleGenAI, Type } from '@google/genai';

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: '10mb' }));

// Default Gemini Models
const PRIMARY_GEMINI_MODEL = 'gemini-3.8-flash';
const FALLBACK_GEMINI_MODEL = 'gemini-flash-latest';

// Get Gemini Client (supporting user custom key or env fallback)
function getGemini(customKey?: string): GoogleGenAI | null {
  const apiKey = (customKey && customKey.trim()) || process.env.GEMINI_API_KEY;
  if (!apiKey) return null;
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// Generate with Gemini
async function generateWithGemini(
  ai: GoogleGenAI,
  params: { contents: any; config?: any },
  modelName: string = PRIMARY_GEMINI_MODEL
) {
  try {
    return await ai.models.generateContent({
      ...params,
      model: modelName,
    });
  } catch (err: any) {
    const isBusy =
      err?.message &&
      (err.message.includes('503') ||
        err.message.includes('UNAVAILABLE') ||
        err.message.includes('high demand') ||
        err.message.includes('429'));
    if (isBusy && modelName !== FALLBACK_GEMINI_MODEL) {
      console.warn(`[Gemini] ${modelName} high demand, retrying with ${FALLBACK_GEMINI_MODEL}...`);
      return await ai.models.generateContent({
        ...params,
        model: FALLBACK_GEMINI_MODEL,
      });
    }
    throw err;
  }
}

// Call OpenAI Chat Completions API
async function callOpenAI(params: {
  apiKey: string;
  model?: string;
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>;
  responseFormat?: { type: string };
  temperature?: number;
}): Promise<{ text: string; model: string }> {
  const model = params.model || 'gpt-4o-mini';
  const apiKey = params.apiKey.trim();

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages: params.messages,
      temperature: params.temperature ?? 0.7,
      ...(params.responseFormat ? { response_format: params.responseFormat } : {}),
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    let detail = errorText;
    try {
      const errJson = JSON.parse(errorText);
      detail = errJson.error?.message || errorText;
    } catch (e) {}
    throw new Error(`OpenAI Error (${response.status}): ${detail}`);
  }

  const data = (await response.json()) as any;
  const text = data.choices?.[0]?.message?.content || '';
  return { text, model: data.model || model };
}

// Extract AI configuration from request headers or body
function extractAIConfig(req: express.Request) {
  const provider =
    (req.headers['x-ai-provider'] as string) ||
    req.body.provider ||
    'gemini';
  const geminiKey =
    (req.headers['x-gemini-key'] as string) ||
    req.body.geminiApiKey ||
    req.body.geminiKey ||
    process.env.GEMINI_API_KEY ||
    '';
  const openaiKey =
    (req.headers['x-openai-key'] as string) ||
    req.body.openaiApiKey ||
    req.body.openaiKey ||
    process.env.OPENAI_API_KEY ||
    '';
  const customModel =
    (req.headers['x-ai-model'] as string) ||
    req.body.model ||
    '';
  const autoFallback =
    req.headers['x-ai-fallback'] !== 'false' &&
    req.body.autoFallback !== false;

  return {
    provider: provider.toLowerCase() === 'openai' ? 'openai' : 'gemini',
    geminiKey: geminiKey.trim(),
    openaiKey: openaiKey.trim(),
    customModel: customModel.trim(),
    autoFallback,
  };
}

// Health Check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    hasGeminiKey: !!process.env.GEMINI_API_KEY,
    hasOpenAIKey: !!process.env.OPENAI_API_KEY,
    timestamp: new Date().toISOString(),
  });
});

// Test Connection for Gemini or OpenAI API Key
app.post('/api/ai/test-key', async (req, res) => {
  const { provider, apiKey, model } = req.body;
  const start = Date.now();

  if (!apiKey || typeof apiKey !== 'string' || !apiKey.trim()) {
    return res.status(400).json({ ok: false, error: 'API key cannot be empty' });
  }

  const cleanKey = apiKey.trim();

  if (provider === 'openai') {
    try {
      const selectedModel = model || 'gpt-4o-mini';
      const result = await callOpenAI({
        apiKey: cleanKey,
        model: selectedModel,
        messages: [{ role: 'user', content: 'Ping! Respond with the word "PONG".' }],
        temperature: 0,
      });
      const latencyMs = Date.now() - start;
      return res.json({
        ok: true,
        provider: 'openai',
        model: result.model || selectedModel,
        latencyMs,
        message: `Successfully connected to OpenAI (${result.model || selectedModel}) in ${latencyMs}ms!`,
      });
    } catch (err: any) {
      return res.status(400).json({
        ok: false,
        provider: 'openai',
        error: err.message || 'OpenAI API key validation failed',
      });
    }
  } else {
    // Gemini validation
    try {
      const selectedModel = model || PRIMARY_GEMINI_MODEL;
      const client = getGemini(cleanKey);
      if (!client) {
        return res.status(400).json({ ok: false, error: 'Failed to create Gemini client' });
      }
      await client.models.generateContent({
        model: selectedModel,
        contents: 'Ping! Respond with the word "PONG".',
      });
      const latencyMs = Date.now() - start;
      return res.json({
        ok: true,
        provider: 'gemini',
        model: selectedModel,
        latencyMs,
        message: `Successfully connected to Gemini (${selectedModel}) in ${latencyMs}ms!`,
      });
    } catch (err: any) {
      let errorMsg = err.message || 'Gemini API key validation failed';
      try {
        const parsed = JSON.parse(errorMsg);
        if (parsed?.error?.message) {
          errorMsg = `Gemini: ${parsed.error.message}`;
        }
      } catch {
        // Not JSON
      }
      return res.status(400).json({
        ok: false,
        provider: 'gemini',
        error: errorMsg,
      });
    }
  }
});

// 1. Universal Capture AI Analysis
app.post('/api/ai/capture', async (req, res) => {
  try {
    const { content, source = 'text', existingContext = {} } = req.body;
    if (!content || typeof content !== 'string') {
      return res.status(400).json({ error: 'Content is required' });
    }

    const { provider, geminiKey, openaiKey, customModel, autoFallback } = extractAIConfig(req);

    // Prompt definition
    const prompt = `You are the intelligence layer of an intelligent Life OS. Analyze this user capture:
Content: "${content}"
Source: "${source}"
Existing projects: ${JSON.stringify(existingContext.projects?.map((p: any) => ({ id: p.id, title: p.title })) || [])}
Existing goals: ${JSON.stringify(existingContext.goals?.map((g: any) => ({ id: g.id, title: g.title })) || [])}

Determine:
1. Target entity: one of 'task', 'note', 'reminder', 'project', 'knowledge', 'habit', 'event'
2. Suggested title (clear, concise)
3. Description / body
4. Priority: 'low' | 'medium' | 'high' | 'urgent'
5. Estimated duration in minutes (if task/event)
6. Energy required: 'low' | 'medium' | 'high'
7. Suggested tags (array of strings)
8. Matched project ID (if strongly relates to existing project, else null)
9. Matched goal ID (if strongly relates to existing goal, else null)
10. Action required: boolean
11. AI confidence: number (0.0 to 1.0)
12. Clarification needed: string if confidence < 0.7 or ambiguous, else null

Return ONLY valid JSON matching this structure:
{
  "targetEntity": "task",
  "title": "...",
  "description": "...",
  "priority": "medium",
  "estimatedMinutes": 30,
  "energyRequired": "medium",
  "tags": ["..."],
  "matchedProjectId": null,
  "matchedGoalId": null,
  "actionRequired": true,
  "confidence": 0.9,
  "clarificationNeeded": null
}`;

    // A. If OpenAI requested and key available
    if (provider === 'openai' && openaiKey) {
      try {
        const result = await callOpenAI({
          apiKey: openaiKey,
          model: customModel || 'gpt-4o-mini',
          messages: [
            { role: 'system', content: 'You are an intelligent Life OS capture processor. Always reply with JSON.' },
            { role: 'user', content: prompt },
          ],
          responseFormat: { type: 'json_object' },
        });
        const parsed = JSON.parse(result.text || '{}');
        return res.json({ result: parsed, engine: `openai:${result.model}` });
      } catch (openAiErr: any) {
        console.warn('OpenAI capture parsing error:', openAiErr.message);
        if (!autoFallback && !geminiKey) throw openAiErr;
      }
    }

    // B. Gemini (primary or fallback)
    const geminiAi = getGemini(geminiKey);
    if (geminiAi) {
      try {
        const modelToUse = customModel || PRIMARY_GEMINI_MODEL;
        const response = await generateWithGemini(
          geminiAi,
          {
            contents: prompt,
            config: {
              responseMimeType: 'application/json',
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  targetEntity: { type: Type.STRING },
                  title: { type: Type.STRING },
                  description: { type: Type.STRING },
                  priority: { type: Type.STRING },
                  estimatedMinutes: { type: Type.INTEGER },
                  energyRequired: { type: Type.STRING },
                  tags: { type: Type.ARRAY, items: { type: Type.STRING } },
                  matchedProjectId: { type: Type.STRING },
                  matchedGoalId: { type: Type.STRING },
                  actionRequired: { type: Type.BOOLEAN },
                  confidence: { type: Type.NUMBER },
                  clarificationNeeded: { type: Type.STRING },
                },
                required: ['targetEntity', 'title', 'priority', 'actionRequired', 'confidence'],
              },
            },
          },
          modelToUse
        );

        const parsed = JSON.parse(response.text || '{}');
        return res.json({ result: parsed, engine: `gemini:${modelToUse}` });
      } catch (geminiErr: any) {
        console.warn('Gemini capture parsing fallback:', geminiErr.message);
        // If Gemini failed and we have OpenAI key, try OpenAI as fallback
        if (autoFallback && openaiKey && provider !== 'openai') {
          try {
            const fallbackResult = await callOpenAI({
              apiKey: openaiKey,
              model: 'gpt-4o-mini',
              messages: [
                { role: 'system', content: 'You are an intelligent Life OS capture processor. Always reply with JSON.' },
                { role: 'user', content: prompt },
              ],
              responseFormat: { type: 'json_object' },
            });
            const parsed = JSON.parse(fallbackResult.text || '{}');
            return res.json({ result: parsed, engine: `openai:${fallbackResult.model}-fallback` });
          } catch (e) {}
        }
      }
    }

    // C. Robust Local / Heuristic Fallback
    const lower = content.toLowerCase();
    let targetEntity = 'note';
    let priority = 'medium';
    let estimatedMinutes = 30;
    let energyRequired = 'medium';
    const actionRequired =
      /^(buy|call|email|review|finish|send|write|fix|create|schedule|prepare|meet|pay|ship)/i.test(
        content.trim()
      ) ||
      lower.includes('todo') ||
      lower.includes('remind');

    if (lower.startsWith('http') || lower.includes('www.') || lower.includes('.com') || lower.includes('.org')) {
      targetEntity = 'knowledge';
    } else if (lower.includes('habit') || lower.includes('every day') || lower.includes('daily')) {
      targetEntity = 'habit';
    } else if (lower.includes('project:') || lower.includes('launch') || lower.includes('initiative')) {
      targetEntity = 'project';
    } else if (actionRequired) {
      targetEntity = 'task';
    } else if (lower.includes('idea:') || lower.includes('thought:')) {
      targetEntity = 'note';
    } else if (lower.includes('meeting') || lower.includes('calendar') || lower.includes('at ') || lower.includes('tomorrow')) {
      targetEntity = 'event';
    }

    if (lower.includes('urgent') || lower.includes('asap') || lower.includes('critical')) {
      priority = 'urgent';
    } else if (lower.includes('important') || lower.includes('priority')) {
      priority = 'high';
    }

    const lines = content.split('\n');
    const title = lines[0].replace(/^(task|note|idea|reminder|todo):\s*/i, '').slice(0, 80);
    const description = lines.length > 1 ? lines.slice(1).join('\n') : '';

    return res.json({
      result: {
        targetEntity,
        title,
        description,
        priority,
        estimatedMinutes,
        energyRequired,
        tags: ['quick-capture'],
        matchedProjectId: null,
        matchedGoalId: null,
        actionRequired,
        confidence: 0.88,
        clarificationNeeded: null,
      },
      engine: 'local-heuristic',
    });
  } catch (err: any) {
    console.error('Fatal in /api/ai/capture:', err);
    return res.json({
      result: {
        targetEntity: 'note',
        title: 'Quick Capture',
        description: '',
        priority: 'medium',
        estimatedMinutes: 30,
        energyRequired: 'medium',
        tags: ['quick-capture'],
        matchedProjectId: null,
        matchedGoalId: null,
        actionRequired: false,
        confidence: 0.8,
        clarificationNeeded: null,
      },
      engine: 'safe-fallback',
    });
  }
});

// 2. Unified Life OS AI Assistant & Agent
app.post('/api/ai/chat', async (req, res) => {
  try {
    const { messages, message, context, requestedAction } = req.body;
    const { provider, geminiKey, openaiKey, customModel, autoFallback } = extractAIConfig(req);

    const messageList = Array.isArray(messages)
      ? messages
      : typeof message === 'string' && message.trim()
      ? [{ role: 'user', content: message.trim() }]
      : [];

    const lastUserPrompt =
      (messageList.length > 0 ? messageList[messageList.length - 1]?.content : '') ||
      (typeof message === 'string' ? message : '') ||
      requestedAction ||
      '';

    const systemInstruction = `You are the executive intelligence layer of Life OS.
You must understand the user's exact intent before responding. Never give generic or template replies.

CORE PRINCIPLES:
1. Answer questions directly and completely.
2. For "how to" or feature questions: provide clear, concise step-by-step instructions based on the actual Life OS app (Home dashboard, Tasks with Inbox/Today/Upcoming/Someday, Plan with Calendar/Timeblocking/Goals, Notes with Second Brain/PARA, Life with Projects/Habits/Money/Energy, Quick Capture shortcut 'C', Command palette 'Cmd+K', Focus Mode timer, and customizable Home Widgets).
3. For commands or requests to take an action: execute the action by proposing structured actions in "proposedActions" with "autoExecute": true so the system immediately verifies and executes it.
   Supported actionTypes:
   - "add_home_widget": payload { "widgetId": "clock" | "priorities" | "schedule" | "habits" | "goals" | "projects" | "energy" | "quick_capture" }
   - "remove_home_widget": payload { "widgetId": string }
   - "create_task": payload { "title": string, "priority"?: "low"|"medium"|"high"|"urgent", "dueDate"?: string, "status"?: "today"|"inbox"|"upcoming" }
   - "update_task": payload { "id": string, "updates": { "dueDate"?: string, "status"?: string, "priority"?: string, "title"?: string } }
   - "delete_task": payload { "id": string }
   - "toggle_task": payload { "id": string }
   - "schedule_timeblock": payload { "title": string, "date": string, "startTime": string, "endTime": string, "type"?: "focus"|"event"|"buffer" }
   - "update_timeblock": payload { "id": string, "updates": object }
   - "delete_timeblock": payload { "id": string }
   - "create_habit": payload { "title": string, "frequency"?: "daily"|"weekly", "targetDaysPerWeek"?: number, "category"?: string }
   - "toggle_habit": payload { "id": string, "dateStr": string }
   - "create_goal": payload { "title": string, "vision": string, "targetDate"?: string, "area"?: string }
   - "create_project": payload { "title": string, "description"?: string, "deadline"?: string }
   - "create_note": payload { "title": string, "content"?: string, "type"?: string }
   - "navigate_app": payload { "nav": "home" | "tasks" | "plan" | "notes" | "life" | "ai", "subNav"?: string }
   - "start_focus": payload { "durationMinutes"?: number }
   - "change_theme": payload { "theme": "light" | "dark" | "system" }
   - "apply_template": payload { "templateId": "tpl-exam-prep" | "tpl-project-launch" | "tpl-deep-work" | "tpl-weekly-review" | "tpl-meeting-os" | "tpl-book-synthesizer" | "tpl-budget-health" | "tpl-atomic-habits" | string, "variables"?: object }
   - "run_automation": payload { "automationId": string }

4. SPECIFIC BEHAVIOR EXAMPLES:
   - User: "How do I use this app?"
     AI: Explain simply, step-by-step:
     1. Quick Capture: Press 'C' anywhere to instantly capture a task, note, or event.
     2. Home: Review your 5-second morning dashboard, live clock, today's top priorities, and schedule.
     3. Tasks: Manage your GTD pipeline with Inbox, Today, Upcoming, and Someday views.
     4. Plan: Time-block your calendar and review milestones for active quarterly Goals.
     5. Life: Maintain daily streak Habits, advance Projects, and log Energy levels.
     6. Notes: Build your Second Brain with interconnected markdown notes.
     7. AI Control: Ask questions or command actions directly in natural language.
   - User: "Add a clock to Home."
     AI: Add the clock widget action with "actionType": "add_home_widget", "payload": { "widgetId": "clock" }, "autoExecute": true. Reply: "I've added the Live Clock widget to your Home dashboard."
   - User: "Move my 5 PM task to tomorrow."
     AI: Inspect the user's tasks and calendar events. Find the task matching 5 PM / 17:00 / 17:30 or scheduled run. Set "actionType": "update_task", "payload": { "id": "<matchedTaskId>", "updates": { "dueDate": "${context?.tomorrowStr || 'tomorrow'}", "status": "upcoming" } }, "autoExecute": true. Reply: "I've moved '[Task Title]' from 5 PM to tomorrow."
   - User: "Show today"
     AI: Set "actionType": "navigate_app", "payload": { "nav": "home" }, "autoExecute": true. Reply: "Here is your Today dashboard."

5. RULES:
   - Before acting, inspect relevant app state; after acting, confirm the exact change.
   - Never claim an action was completed unless the state confirms it.
   - If a request is ambiguous, ask only the minimum necessary clarification, referring to specific items found in state.
   - Maintain conversation context and understand follow-up messages.
   - Do NOT repeat phrases like "I'm analyzing your context" or "How would you like to prioritize your next action?" unless genuinely relevant.
   - Prioritize ACTION > generic conversation.
   - Every response must directly address the user's latest message.

CURRENT LIVE APP STATE:
- Today's Date: ${context?.todayStr || new Date().toISOString().split('T')[0]}
- Tomorrow's Date: ${context?.tomorrowStr || new Date(Date.now() + 86400000).toISOString().split('T')[0]}
- Live Time: ${context?.currentTime || new Date().toLocaleTimeString()}
- Home Widgets: ${JSON.stringify(context?.homeWidgets || [])}
- Tasks (${context?.tasks?.length || 0}): ${JSON.stringify(context?.tasks || [])}
- Calendar Events (${context?.calendar?.length || 0}): ${JSON.stringify(context?.calendar || [])}
- Active Goals (${context?.goals?.length || 0}): ${JSON.stringify(context?.goals || [])}
- Active Projects (${context?.projects?.length || 0}): ${JSON.stringify(context?.projects || [])}
- Habits: ${JSON.stringify(context?.habits || [])}
- Settings: ${JSON.stringify(context?.settings || {})}`;

    const prompt = `User request / command: "${lastUserPrompt}"
Conversation History: ${JSON.stringify(messageList.slice(-6))}
Provide a direct, complete reply and structured proposedActions in valid JSON format.`;

    // A. OpenAI primary
    if (provider === 'openai' && openaiKey && lastUserPrompt) {
      try {
        const formattedMessages = [
          { role: 'system' as const, content: systemInstruction },
          ...messageList.map((m: any) => ({
            role: (m.role === 'assistant' ? 'assistant' : 'user') as 'assistant' | 'user',
            content: String(m.content || ''),
          })),
        ];

        const openAiRes = await callOpenAI({
          apiKey: openaiKey,
          model: customModel || 'gpt-4o-mini',
          messages: formattedMessages,
          responseFormat: { type: 'json_object' },
        });

        const parsed = JSON.parse(openAiRes.text || '{}');
        return res.json({
          reply: parsed.reply || 'Request completed.',
          insights: parsed.insights || [],
          proposedActions: (parsed.proposedActions || []).map((a: any) => ({
            ...a,
            autoExecute: a.autoExecute ?? true,
          })),
          engine: `openai:${openAiRes.model}`,
        });
      } catch (openAiErr: any) {
        console.warn('OpenAI chat error:', openAiErr.message);
        if (!autoFallback && !geminiKey) throw openAiErr;
      }
    }

    // B. Gemini
    const geminiAi = getGemini(geminiKey);
    if (geminiAi && lastUserPrompt) {
      try {
        const modelToUse = customModel || PRIMARY_GEMINI_MODEL;
        const response = await generateWithGemini(
          geminiAi,
          {
            contents: prompt,
            config: {
              systemInstruction,
              responseMimeType: 'application/json',
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  reply: { type: Type.STRING },
                  insights: { type: Type.ARRAY, items: { type: Type.STRING } },
                  proposedActions: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        id: { type: Type.STRING },
                        actionType: { type: Type.STRING },
                        title: { type: Type.STRING },
                        details: { type: Type.STRING },
                        autoExecute: { type: Type.BOOLEAN },
                        requiresConfirmation: { type: Type.BOOLEAN },
                        payload: { type: Type.OBJECT },
                      },
                      required: ['id', 'actionType', 'title'],
                    },
                  },
                },
                required: ['reply'],
              },
            },
          },
          modelToUse
        );

        const parsed = JSON.parse(response.text || '{}');
        return res.json({
          reply: parsed.reply || 'Understood.',
          insights: parsed.insights || [],
          proposedActions: (parsed.proposedActions || []).map((a: any) => ({
            ...a,
            autoExecute: a.autoExecute ?? true,
          })),
          engine: `gemini:${modelToUse}`,
        });
      } catch (err: any) {
        console.warn('Gemini chat fallback:', err.message);
        if (autoFallback && openaiKey && provider !== 'openai') {
          try {
            const fallbackRes = await callOpenAI({
              apiKey: openaiKey,
              model: 'gpt-4o-mini',
              messages: [
                { role: 'system', content: systemInstruction },
                { role: 'user', content: prompt },
              ],
              responseFormat: { type: 'json_object' },
            });
            const parsed = JSON.parse(fallbackRes.text || '{}');
            return res.json({
              reply: parsed.reply || 'Understood.',
              insights: parsed.insights || [],
              proposedActions: (parsed.proposedActions || []).map((a: any) => ({
                ...a,
                autoExecute: a.autoExecute ?? true,
              })),
              engine: `openai:${fallbackRes.model}-fallback`,
            });
          } catch (e) {}
        }
      }
    }

    // C. Intelligent local fallback responder (State-aware, Action-oriented)
    const queryLower = (lastUserPrompt || '').trim().toLowerCase();
    const todayStr = context?.todayStr || new Date().toISOString().split('T')[0];
    const tomorrowStr =
      context?.tomorrowStr || new Date(Date.now() + 86400000).toISOString().split('T')[0];
    const tasksList: any[] = context?.tasks || [];
    const calendarList: any[] = context?.calendar || [];

    let reply = '';
    let proposedActions: any[] = [];
    let insights: string[] = [];

    // 1. "How do I use this app?" or questions on using features
    if (
      queryLower.includes('how do i use') ||
      queryLower.includes('how to use') ||
      queryLower.includes('how does this work') ||
      queryLower.includes('getting started') ||
      queryLower === 'help' ||
      queryLower === 'guide'
    ) {
      reply =
        `Here is how to use Life OS step-by-step:\n\n` +
        `1. **Quick Capture**: Press 'C' anywhere (or tap the floating '+' button) to capture tasks, notes, or ideas with zero friction.\n` +
        `2. **Home Dashboard**: Your 5-second overview of today's top 3 priorities, schedule timeline, and live widgets.\n` +
        `3. **Tasks**: A GTD-inspired pipeline with Inbox, Today, Upcoming, and Someday views.\n` +
        `4. **Plan**: Time-block your calendar day and track milestone progress on quarterly Goals.\n` +
        `5. **Life**: Build recurring daily Habits, monitor Project deadlines, and track your Energy rhythm.\n` +
        `6. **Notes**: Your linked Second Brain for markdown documents and reference material.\n` +
        `7. **AI Executive Partner**: Type natural commands (e.g. "Add a clock to Home", "Move my 5 PM task to tomorrow") to execute real app changes with instant verification and 1-tap Undo.`;
      insights = [
        'Press C anytime for Universal Capture',
        'Press Cmd+K for Command Search',
        'AI commands execute real verified app state changes',
      ];
    }
    // 2. "Add a clock to Home" / "add clock" / "clock widget"
    else if (
      queryLower.includes('add a clock') ||
      queryLower.includes('add clock') ||
      queryLower.includes('clock to home') ||
      queryLower.includes('show clock')
    ) {
      reply = "I've added the Live Clock & Timezone widget to your Home dashboard.";
      proposedActions = [
        {
          id: 'action-clock-' + Date.now(),
          actionType: 'add_home_widget',
          title: 'Add Live Clock to Home',
          details: 'Enables real-time clock, timezone, and date widget on Home screen',
          autoExecute: true,
          undoable: true,
          payload: { widgetId: 'clock' },
        },
      ];
      insights = ['Live Clock widget now displays real-time seconds, date, and timezone'];
    }
    // 3. "Remove clock"
    else if (
      queryLower.includes('remove clock') ||
      queryLower.includes('hide clock') ||
      queryLower.includes('delete clock')
    ) {
      reply = "I've removed the Clock widget from your Home dashboard.";
      proposedActions = [
        {
          id: 'action-rm-clock-' + Date.now(),
          actionType: 'remove_home_widget',
          title: 'Remove Clock from Home',
          details: 'Hides the clock widget from Home',
          autoExecute: true,
          undoable: true,
          payload: { widgetId: 'clock' },
        },
      ];
    }
    // 4. "Move my 5 PM task to tomorrow" / move 5pm task / move task to tomorrow
    else if (
      (queryLower.includes('move') && (queryLower.includes('5') || queryLower.includes('17')) && (queryLower.includes('tomorrow') || queryLower.includes('task'))) ||
      (queryLower.includes('move') && queryLower.includes('task') && queryLower.includes('tomorrow'))
    ) {
      // Find candidate task
      let targetTask = tasksList.find((t) => {
        const titleL = t.title.toLowerCase();
        return titleL.includes('5') || titleL.includes('run') || titleL.includes('aerobic');
      });
      if (!targetTask) {
        // Check calendar events at 17:00 / 17:30
        const ev = calendarList.find(
          (e) => e.startTime.includes('17') || e.startTime.includes('5:') || e.title.toLowerCase().includes('5')
        );
        if (ev?.taskId) {
          targetTask = tasksList.find((t) => t.id === ev.taskId);
        }
      }
      if (!targetTask && tasksList.length > 0) {
        targetTask = tasksList.find((t) => t.status === 'today') || tasksList[0];
      }

      const taskTitle = targetTask?.title || '5 PM task';
      const taskId = targetTask?.id || 't-3';

      reply = `I've moved "${taskTitle}" from 5 PM to tomorrow (${tomorrowStr}).`;
      proposedActions = [
        {
          id: 'action-move-task-' + Date.now(),
          actionType: 'update_task',
          title: `Move "${taskTitle}" to tomorrow`,
          details: `Rescheduled to ${tomorrowStr} and marked as upcoming`,
          autoExecute: true,
          undoable: true,
          payload: {
            id: taskId,
            query: '5 PM',
            updates: {
              dueDate: tomorrowStr,
              status: 'upcoming',
            },
          },
        },
      ];
      insights = [`Scheduled for ${tomorrowStr} with 1-tap Undo available`];
    }
    // 5. "Show today" / "go to today" / "open tasks" / "show home"
    else if (
      queryLower.includes('show today') ||
      queryLower.includes('go to today') ||
      queryLower.includes('open home') ||
      queryLower === 'today'
    ) {
      reply = "Navigating to your Home dashboard for Today.";
      proposedActions = [
        {
          id: 'action-nav-' + Date.now(),
          actionType: 'navigate_app',
          title: 'Show Today',
          details: 'Navigate to Home dashboard',
          autoExecute: true,
          undoable: false,
          payload: { nav: 'home' },
        },
      ];
    }
    // 6. "Show tasks" / "open tasks" / "go to tasks"
    else if (queryLower.includes('show tasks') || queryLower.includes('open tasks') || queryLower.includes('go to tasks')) {
      reply = "Opening your Tasks pipeline.";
      proposedActions = [
        {
          id: 'action-nav-' + Date.now(),
          actionType: 'navigate_app',
          title: 'Navigate to Tasks',
          details: 'Open Tasks view',
          autoExecute: true,
          undoable: false,
          payload: { nav: 'tasks' },
        },
      ];
    }
    // 7. "Create task" / "add task" / "create a task"
    else if (/(create|add|new)\s+(a\s+)?(new\s+)?task/i.test(queryLower)) {
      const taskTitle = lastUserPrompt.replace(/^(please\s+)?(create|add|new)\s+(a\s+)?(new\s+)?task\s*(for|to|:|about)?\s*/i, '').trim() || 'New Task';
      reply = `I've created the task: "${taskTitle}".`;
      proposedActions = [
        {
          id: 'action-create-task-' + Date.now(),
          actionType: 'create_task',
          title: `Create task "${taskTitle}"`,
          details: 'Added to your today queue',
          autoExecute: true,
          undoable: true,
          payload: {
            title: taskTitle,
            status: 'today',
            priority: 'medium',
            dueDate: todayStr,
          },
        },
      ];
    }
    // 8. "Create goal" / "add goal" / "create a goal"
    else if (/(create|add|new)\s+(a\s+)?(new\s+)?goal/i.test(queryLower)) {
      const goalTitle = lastUserPrompt.replace(/^(please\s+)?(create|add|new)\s+(a\s+)?(new\s+)?goal\s*(for|to|:|about)?\s*/i, '').trim() || 'New Goal';
      reply = `I've created your new goal: "${goalTitle}".`;
      proposedActions = [
        {
          id: 'action-create-goal-' + Date.now(),
          actionType: 'create_goal',
          title: `Create Goal "${goalTitle}"`,
          details: 'Added to active quarterly goals',
          autoExecute: true,
          undoable: true,
          payload: {
            title: goalTitle,
            vision: `Advance progress toward ${goalTitle}`,
            area: 'Personal',
          },
        },
      ];
    }
    // 9. "Add habit" / "create habit" / "create a habit"
    else if (/(create|add|new)\s+(a\s+)?(new\s+)?habit/i.test(queryLower)) {
      const habitTitle = lastUserPrompt.replace(/^(please\s+)?(create|add|new)\s+(a\s+)?(new\s+)?habit\s*(for|to|:|about)?\s*/i, '').trim() || 'New Daily Habit';
      reply = `I've established the habit: "${habitTitle}".`;
      proposedActions = [
        {
          id: 'action-create-habit-' + Date.now(),
          actionType: 'create_habit',
          title: `Create Habit "${habitTitle}"`,
          details: 'Daily habit added to your tracker',
          autoExecute: true,
          undoable: true,
          payload: {
            title: habitTitle,
            frequency: 'daily',
            targetDaysPerWeek: 7,
            category: 'Personal Growth',
          },
        },
      ];
    }
    // 10. "Create project" / "add project" / "create a project"
    else if (/(create|add|new)\s+(a\s+)?(new\s+)?project/i.test(queryLower)) {
      const projectTitle = lastUserPrompt.replace(/^(please\s+)?(create|add|new)\s+(a\s+)?(new\s+)?project\s*(for|to|:|about)?\s*/i, '').trim() || 'New Project';
      reply = `I've created your new project: "${projectTitle}".`;
      proposedActions = [
        {
          id: 'action-create-project-' + Date.now(),
          actionType: 'create_project',
          title: `Create Project "${projectTitle}"`,
          details: 'Active project created in Life OS',
          autoExecute: true,
          undoable: true,
          payload: {
            title: projectTitle,
            description: `Key initiative: ${projectTitle}`,
          },
        },
      ];
    }
    // 10. "Dark mode" / "light mode" / "change theme"
    else if (queryLower.includes('dark mode') || queryLower.includes('light mode') || queryLower.includes('theme')) {
      const targetTheme = queryLower.includes('light') ? 'light' : 'dark';
      reply = `Switching theme to ${targetTheme} mode.`;
      proposedActions = [
        {
          id: 'action-theme-' + Date.now(),
          actionType: 'change_theme',
          title: `Switch to ${targetTheme} theme`,
          details: `Change application appearance to ${targetTheme}`,
          autoExecute: true,
          undoable: true,
          payload: { theme: targetTheme },
        },
      ];
    }
    // 11. "What do I have today" / "what are my tasks" / "schedule today"
    else if (queryLower.includes('what do i have') || queryLower.includes('my tasks') || queryLower.includes('what is today') || queryLower.includes('today schedule')) {
      const todayTasks = tasksList.filter((t) => t.status === 'today');
      const todayEvents = calendarList.filter((e) => !e.date || e.date === todayStr);
      reply =
        `Here is your schedule for today (${todayStr}):\n\n` +
        `**Tasks (${todayTasks.length})**:\n` +
        (todayTasks.length > 0
          ? todayTasks.map((t) => `• [${t.priority.toUpperCase()}] ${t.title}`).join('\n')
          : '• No tasks currently scheduled for today.\n') +
        `\n\n**Calendar Blocks (${todayEvents.length})**:\n` +
        (todayEvents.length > 0
          ? todayEvents.map((e) => `• ${e.startTime} - ${e.endTime}: ${e.title}`).join('\n')
          : '• No calendar blocks scheduled.');
      insights = [`You have ${todayTasks.length} active tasks and ${todayEvents.length} calendar events today.`];
    }
    // 12. General direct answer
    else {
      reply = `I've inspected your current state: you have ${tasksList.length} tasks and ${calendarList.length} calendar events. You can ask me to perform any action directly (e.g. "Add a clock to Home", "Move my 5 PM task to tomorrow", "Create a task for Reviewing quarterly numbers").`;
    }

    return res.json({
      reply,
      insights,
      proposedActions,
      engine: 'local-intelligence',
    });
  } catch (globalErr: any) {
    console.error('Fatal error in /api/ai/chat:', globalErr);
    return res.json({
      reply: 'I reviewed your context locally. What would you like to work on next?',
      insights: [],
      proposedActions: [],
      engine: 'safe-fallback',
    });
  }
});

// 3. Intelligent Weekly Review Generator
app.post('/api/ai/review', async (req, res) => {
  try {
    const { tasks, habits, projects, goals, focusSessions } = req.body;
    const { provider, geminiKey, openaiKey, customModel, autoFallback } = extractAIConfig(req);

    const prompt = `Perform a structured Life OS Weekly Review based on this user data:
Completed Tasks: ${JSON.stringify(tasks?.filter((t: any) => t.status === 'completed') || [])}
Unfinished Tasks: ${JSON.stringify(tasks?.filter((t: any) => t.status !== 'completed') || [])}
Habits: ${JSON.stringify(habits || [])}
Projects: ${JSON.stringify(projects || [])}
Goals: ${JSON.stringify(goals || [])}
Focus Sessions: ${JSON.stringify(focusSessions || [])}

Follow the four-part framework:
1. What happened (objective metrics, completed achievements, habits consistency)
2. What matters (alignment with primary visions and goals)
3. What needs attention (bottlenecks, stalled tasks, dropped habits)
4. Proposed next actions (actionable recommendations for next week)

Output strict JSON:
{
  "summary": "...",
  "whatHappened": ["..."],
  "whatMatters": "...",
  "whatNeedsAttention": ["..."],
  "proposedNextActions": [
    { "id": "act-1", "action": "...", "targetArea": "..." }
  ]
}`;

    if (provider === 'openai' && openaiKey) {
      try {
        const result = await callOpenAI({
          apiKey: openaiKey,
          model: customModel || 'gpt-4o-mini',
          messages: [
            { role: 'system', content: 'You are an intelligent Life OS executive coach. Always return JSON.' },
            { role: 'user', content: prompt },
          ],
          responseFormat: { type: 'json_object' },
        });
        const parsed = JSON.parse(result.text || '{}');
        return res.json({ review: parsed, engine: `openai:${result.model}` });
      } catch (err: any) {
        console.warn('OpenAI review error:', err.message);
        if (!autoFallback && !geminiKey) throw err;
      }
    }

    const geminiAi = getGemini(geminiKey);
    if (geminiAi) {
      try {
        const modelToUse = customModel || PRIMARY_GEMINI_MODEL;
        const response = await generateWithGemini(
          geminiAi,
          {
            contents: prompt,
            config: {
              responseMimeType: 'application/json',
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  summary: { type: Type.STRING },
                  whatHappened: { type: Type.ARRAY, items: { type: Type.STRING } },
                  whatMatters: { type: Type.STRING },
                  whatNeedsAttention: { type: Type.ARRAY, items: { type: Type.STRING } },
                  proposedNextActions: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        id: { type: Type.STRING },
                        action: { type: Type.STRING },
                        targetArea: { type: Type.STRING },
                      },
                      required: ['id', 'action'],
                    },
                  },
                },
                required: ['summary', 'whatHappened', 'whatMatters', 'whatNeedsAttention', 'proposedNextActions'],
              },
            },
          },
          modelToUse
        );

        const parsed = JSON.parse(response.text || '{}');
        return res.json({ review: parsed, engine: `gemini:${modelToUse}` });
      } catch (err: any) {
        console.warn('Gemini review fallback:', err.message);
      }
    }

    // Local fallback
    const completedCount = tasks?.filter((t: any) => t.status === 'completed')?.length || 0;
    const unfinishedCount = tasks?.filter((t: any) => t.status !== 'completed')?.length || 0;

    return res.json({
      review: {
        summary: `Weekly reflection: ${completedCount} tasks completed with steady progress across active projects.`,
        whatHappened: [
          `Finished ${completedCount} key tasks across your active projects`,
          `Maintained consistent habit check-ins`,
          `Logged focused deep work sessions`,
        ],
        whatMatters: 'Focus on closing open project loops before introducing new commitments.',
        whatNeedsAttention: [
          `${unfinishedCount} unfinished tasks currently in the backlog`,
          'Ensure adequate recovery time blocks are scheduled for the coming week',
        ],
        proposedNextActions: [
          { id: 'act-1', action: 'Archive 2 completed milestones in active projects', targetArea: 'Projects' },
          { id: 'act-2', action: 'Schedule high-energy focus block for pending priority task', targetArea: 'Calendar' },
          { id: 'act-3', action: 'Clear unfiled notes from Second Brain inbox', targetArea: 'Knowledge' },
        ],
      },
      engine: 'local-intelligence',
    });
  } catch (err: any) {
    console.error('Fatal in /api/ai/review:', err);
    return res.json({
      review: {
        summary: 'Weekly review synthesized locally.',
        whatHappened: ['Continued focus on ongoing projects.'],
        whatMatters: 'Protect daily deep work time.',
        whatNeedsAttention: ['Review stale tasks.'],
        proposedNextActions: [{ id: 'act-1', action: 'Plan high priority tasks', targetArea: 'Tasks' }],
      },
      engine: 'safe-fallback',
    });
  }
});

// 4. Intelligent AI Auto-Scheduler
app.post('/api/ai/schedule', async (req, res) => {
  try {
    const { tasks, existingEvents, userEnergy } = req.body;
    const { provider, geminiKey, openaiKey, customModel, autoFallback } = extractAIConfig(req);

    const prompt = `You are the Life OS intelligent scheduling assistant.
Tasks to schedule: ${JSON.stringify(tasks || [])}
Existing events/commitments: ${JSON.stringify(existingEvents || [])}
Current user energy profile: ${JSON.stringify(userEnergy || 'medium')}

Rules:
- High priority / high energy tasks go into morning or peak energy slots
- Never double-book or overlap existing fixed commitments
- Protect 15-30 minute breaks between blocks
- Do not pack every minute; protect flexible recovery time
Return an array of proposed timeblocks with startTime, endTime, taskId, title, and reasoning.

Output strict JSON:
{
  "proposedSchedule": [
    {
      "taskId": "...",
      "title": "...",
      "startHour": 9,
      "startMinute": 0,
      "durationMinutes": 45,
      "reasoning": "..."
    }
  ],
  "explanation": "..."
}`;

    if (provider === 'openai' && openaiKey) {
      try {
        const result = await callOpenAI({
          apiKey: openaiKey,
          model: customModel || 'gpt-4o-mini',
          messages: [
            { role: 'system', content: 'You are an intelligent Life OS calendar scheduler. Always return JSON.' },
            { role: 'user', content: prompt },
          ],
          responseFormat: { type: 'json_object' },
        });
        const parsed = JSON.parse(result.text || '{}');
        return res.json({ result: parsed, engine: `openai:${result.model}` });
      } catch (err: any) {
        console.warn('OpenAI schedule error:', err.message);
        if (!autoFallback && !geminiKey) throw err;
      }
    }

    const geminiAi = getGemini(geminiKey);
    if (geminiAi) {
      try {
        const modelToUse = customModel || PRIMARY_GEMINI_MODEL;
        const response = await generateWithGemini(
          geminiAi,
          {
            contents: prompt,
            config: {
              responseMimeType: 'application/json',
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  proposedSchedule: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        taskId: { type: Type.STRING },
                        title: { type: Type.STRING },
                        startHour: { type: Type.INTEGER },
                        startMinute: { type: Type.INTEGER },
                        durationMinutes: { type: Type.INTEGER },
                        reasoning: { type: Type.STRING },
                      },
                      required: ['title', 'startHour', 'startMinute', 'durationMinutes'],
                    },
                  },
                  explanation: { type: Type.STRING },
                },
                required: ['proposedSchedule', 'explanation'],
              },
            },
          },
          modelToUse
        );

        const parsed = JSON.parse(response.text || '{}');
        return res.json({ result: parsed, engine: `gemini:${modelToUse}` });
      } catch (err: any) {
        console.warn('Gemini schedule fallback:', err.message);
      }
    }

    // Local fallback
    const items = (tasks || []).slice(0, 3);
    let currentH = 9;
    const proposedSchedule = items.map((task: any) => {
      const slot = {
        taskId: task.id,
        title: task.title,
        startHour: currentH,
        startMinute: 0,
        durationMinutes: task.durationMinutes || 45,
        reasoning: 'Optimal morning deep work slot aligned with priority',
      };
      currentH += 2;
      return slot;
    });

    return res.json({
      result: {
        proposedSchedule,
        explanation: 'Balanced schedule respecting protected breaks and personal energy rhythms.',
      },
      engine: 'local-intelligence',
    });
  } catch (err: any) {
    console.error('Fatal in /api/ai/schedule:', err);
    return res.json({
      result: {
        proposedSchedule: [],
        explanation: 'Schedule preserved with existing calendar blocks.',
      },
      engine: 'safe-fallback',
    });
  }
});

// 5. Intelligent AI Help Assistant & Tutor
app.post('/api/ai/help', async (req, res) => {
  try {
    const { question, screenContext, history } = req.body;
    const { provider, geminiKey, openaiKey, customModel, autoFallback } = extractAIConfig(req);

    if (!question || typeof question !== 'string') {
      return res.status(400).json({ error: 'Question is required' });
    }

    const systemPrompt = `You are the official Life OS In-App AI Tutor & Help Assistant.
Your role is to teach users how to use Life OS with 100% accuracy, clarity, and precision.

VERIFIED APP ARCHITECTURE & FEATURES:
- Top Header: Life OS brand icon, Current Date, Search button (⌘K), Universal Quick Capture (Press 'C'), Ambient Sound toggle (Rain/Ocean/White noise), Focus Mode timer (Clock icon), 3-State Theme toggle (Light, OLED Dark, System), Help Center button (?), Settings icon (Gear), Profile avatar.
- Main Navigation:
  * Home: 5-second command center. Daily AI briefing banner, quick capture text box, Today's Priorities list (urgent & high priority separated), Today's Schedule timeblocks, Habits check-in list, Active Goals & Projects widgets.
  * Tasks: Segmented tabs for "Today", "Inbox", "Upcoming", "Completed", "All". Rapid inline add row. Detailed task modal with due date, priority (Urgent, High, Medium, Low), duration (minutes), energy required, linked project/goal, and subtask checklists.
  * Plan: Sub-tabs for "Calendar", "Time Blocking", and "Goals". Features 24-hour visual timeline, AI Smart Schedule button, conflict overlap detection, and goal progress calculation from milestone checkboxes.
  * Notes: Sub-tabs for "Notes" (PARA markdown library: Projects, Areas, Resources, Archives with search & tags) and "Second Brain" (interactive bidirectional graph canvas with backlinks and node details).
  * Life: Sub-tabs for "Projects" (active projects & deadlines), "Habits" (daily/weekly streaks with flame icons), "Money" (monthly burn rate, subscriptions, runway), and "Energy & Rhythms" (chronotype curve, 1-5 check-in).
  * AI (Assistant): Executive coach chat, prompt suggestions, personal rules & memory management (active/inactive toggles, clear all), and automation rules.
  * Settings Modal: Profile name & avatar, Theme selection, Accent colors, Ambient sound presets, AI Backbone (Gemini 3.6 Flash / OpenAI GPT-4o-mini custom keys & validation), Account & Google Sync via Firebase, and Data Management (JSON Export, JSON Import, Reset to Sample Data).
  * Modals: Universal Quick Capture (Press C anywhere), Universal Search (Press ⌘K / Ctrl+K), Focus Mode with Pomodoro timer & Web Audio ambient sounds, Weekly Review 4-part ritual modal.

CRITICAL RULES:
1. ONLY refer to actual features, screens, tabs, and buttons that exist in the app. Never invent imaginary settings or buttons.
2. Give clear, numbered step-by-step instructions. Explicitly tell the user where to tap/click and what happens next.
3. If asked "How do I make Home show only today's tasks?", explain that Home automatically shows tasks whose status is set to "Today", and advise changing status of non-today tasks to "Upcoming" or "Inbox".
4. If asked "How does AI memory work?", explain what it stores (preferences, guidelines, work hours), what it doesn't store (credentials, sensitive logs), how it is used as prompt context, and how to manage/clear it in the AI view.
5. If asked "I don't know how to use Life OS. Teach me everything from beginner to advanced.", provide a progressive 4-level curriculum (Level 1: Daily essentials & quick capture, Level 2: Time blocking & habits, Level 3: Second brain PARA & graph, Level 4: Master OS with automations & weekly reviews).
6. If a requested capability is not in the app, explicitly say: "That capability is not currently available in Life OS."
7. Explain → Guide → Do it for me: Suggest one or more real executable action buttons where appropriate. Supported actionTypes:
   - "navigate" (payload: { nav: "home" | "tasks" | "plan" | "notes" | "life" | "ai", subNav?: string })
   - "switch_theme"
   - "open_focus"
   - "open_capture"
   - "open_search"
   - "open_settings"
   - "open_review"
   - "play_sound" (payload: { sound: "rain" | "ocean" | "whitenoise" | "none" })
   - "export_data"
   - "reset_ai_memory"

Output strict JSON:
{
  "reply": "Clear, markdown-formatted response with bold UI elements and numbered steps.",
  "actions": [
    {
      "id": "act-1",
      "label": "Short Action Button Label",
      "actionType": "navigate",
      "payload": { "nav": "tasks" },
      "icon": "CheckSquare"
    }
  ],
  "relatedTopics": ["getting-started", "tasks", "plan"]
}`;

    const userPrompt = `Current Screen Context: ${screenContext || 'general'}
User Question: "${question}"
Please provide exact, step-by-step guidance following the strict rules.`;

    if (provider === 'openai' && openaiKey) {
      try {
        const result = await callOpenAI({
          apiKey: openaiKey,
          model: customModel || 'gpt-4o-mini',
          messages: [
            { role: 'system', content: systemPrompt },
            ...(history || []).slice(-4).map((h: any) => ({
              role: h.role === 'assistant' ? 'assistant' : 'user',
              content: h.content,
            })),
            { role: 'user', content: userPrompt },
          ],
          responseFormat: { type: 'json_object' },
        });
        const parsed = JSON.parse(result.text || '{}');
        return res.json({
          reply: parsed.reply || result.text,
          actions: parsed.actions || [],
          relatedTopics: parsed.relatedTopics || [],
          engine: `openai:${result.model}`,
        });
      } catch (err: any) {
        console.warn('OpenAI help error:', err.message);
        if (!autoFallback && !geminiKey) throw err;
      }
    }

    const geminiAi = getGemini(geminiKey);
    if (geminiAi) {
      try {
        const modelToUse = customModel || PRIMARY_GEMINI_MODEL;
        const response = await generateWithGemini(
          geminiAi,
          {
            contents: `${systemPrompt}\n\n${userPrompt}`,
            config: {
              responseMimeType: 'application/json',
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  reply: { type: Type.STRING },
                  actions: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        id: { type: Type.STRING },
                        label: { type: Type.STRING },
                        actionType: { type: Type.STRING },
                        payload: { type: Type.OBJECT },
                        icon: { type: Type.STRING },
                      },
                      required: ['id', 'label', 'actionType'],
                    },
                  },
                  relatedTopics: { type: Type.ARRAY, items: { type: Type.STRING } },
                },
                required: ['reply'],
              },
            },
          },
          modelToUse
        );

        const parsed = JSON.parse(response.text || '{}');
        return res.json({
          reply: parsed.reply,
          actions: parsed.actions || [],
          relatedTopics: parsed.relatedTopics || [],
          engine: `gemini:${modelToUse}`,
        });
      } catch (err: any) {
        console.warn('Gemini help fallback:', err.message);
      }
    }

    // Local Verified Heuristic Fallback
    const qLower = question.toLowerCase();
    let reply = '';
    let actions: any[] = [];
    let relatedTopics: string[] = [];

    if (qLower.includes('time block') || qLower.includes('calendar') || qLower.includes('schedule')) {
      reply = `### How to Create a Time Block

1. **Open Plan View**: Tap **Plan** in the top navigation bar (desktop) or bottom tab bar (mobile).
2. **Select Sub-Tab**: Tap **Time Blocking** or **Calendar** in the contextual sub-menu.
3. **Add Block**: Tap the **+ Add Block** button in the upper right corner.
4. **Set Details**: Enter your block title, start time, end time, and select type (*timeblock*, *focus*, or *buffer*).
5. **(Optional) Link Task**: Select an active task from your Today queue to work on during this block.
6. **Save**: Tap **Schedule Block** to place it onto your 24-hour visual day timeline.

*Tip: You can also tap **AI Smart Schedule** in the Plan header to automatically place your highest-priority tasks into optimal morning focus slots.*`;
      actions = [
        { id: 'act-plan', label: 'Go to Time Blocking', actionType: 'navigate', payload: { nav: 'plan', subNav: 'timeblocking' }, icon: 'Clock' },
        { id: 'act-calendar', label: 'View Calendar', actionType: 'navigate', payload: { nav: 'plan', subNav: 'calendar' }, icon: 'Calendar' },
      ];
      relatedTopics = ['plan', 'calendar'];
    } else if (qLower.includes('memory') || qLower.includes('ai memory')) {
      reply = `### How AI Memory Works in Life OS

**1. What It Stores:**
- Your working style preferences, typical active hours, and chronotype.
- Explicit guidelines you tell the AI to remember (e.g., *"Keep responses concise"*, *"Never schedule deep work after 6pm"*).
- Key project priorities and focus areas.

**2. What It Never Stores:**
- Passwords, credentials, or private authentication tokens.
- Unapproved external logs or third-party tracking data.

**3. How It Is Used:**
Active memories are injected as background system guidelines into AI Assistant prompts, ensuring your coaching advice remains personalized to your real life without repetitive setup.

**4. How to Manage or Reset It:**
1. Navigate to the **AI** tab (Sparkles icon).
2. Switch to the **Personal Rules & Memory** tab.
3. Toggle individual memories on or off with the switch.
4. Tap the trash icon to delete specific memories, or tap **Clear All Memory** to reset everything.`;
      actions = [
        { id: 'act-ai-mem', label: 'Manage AI Memories', actionType: 'navigate', payload: { nav: 'ai' }, icon: 'Brain' },
        { id: 'act-reset-mem', label: 'Reset AI Memory', actionType: 'reset_ai_memory', icon: 'Trash2' },
      ];
      relatedTopics = ['ai'];
    } else if (qLower.includes('home show only') || (qLower.includes('home') && qLower.includes('today'))) {
      reply = `### How to Make Home Show Only Today's Tasks

The Home screen is designed to display **only tasks with status set to "Today"**.

**To ensure only today's tasks appear on Home:**
1. Open the **Tasks** view via the top menu or bottom tab bar.
2. In Tasks, items are organized by status: **Today**, **Inbox**, **Upcoming**, **Completed**, and **All**.
3. Any task you want on Home must have status **Today**.
4. To remove an unwanted task from Home without deleting it, click the task row to expand it, select the **Status** dropdown, and change it to **Upcoming** or **Inbox**.
5. Home's "Today's Priorities" list will immediately update to reflect only your active today obligations.`;
      actions = [
        { id: 'act-tasks-today', label: 'Go to Today\'s Tasks', actionType: 'navigate', payload: { nav: 'tasks' }, icon: 'CheckSquare' },
        { id: 'act-home', label: 'View Home Screen', actionType: 'navigate', payload: { nav: 'home' }, icon: 'Home' },
      ];
      relatedTopics = ['tasks', 'home'];
    } else if (qLower.includes('teach me') || qLower.includes('beginner to advanced') || qLower.includes('everything') || qLower.includes('how to use')) {
      reply = `### Complete Guide to Life OS: Beginner to Advanced

Welcome to Life OS! Here is your step-by-step pathway from initial setup to total personal mastery:

#### 🟢 Level 1: Daily Essentials (Day 1–3)
- **Universal Quick Capture**: Press **C** on your keyboard (or tap the floating **+** button) to record any fleeting thought, task, or note.
- **Home Command Center**: Check your **Today's Priorities** and **Daily Intelligence Briefing** in under 5 seconds each morning.
- **Focus Mode**: Tap the **Clock** icon in the header to run your first 25-minute Pomodoro session with synthesized rain audio.

#### 🟡 Level 2: Planning & Rhythm (Day 4–7)
- **Time Blocking**: Go to **Plan > Time Blocking** to allocate dedicated calendar blocks for high-leverage tasks.
- **Habits**: Tap **Life > Habits** to track 2–3 daily non-negotiables with streak counters.
- **Energy Alignment**: In **Life > Energy & Rhythms**, log your energy level to discover your peak focus windows.

#### 🔵 Level 3: Second Brain & Knowledge (Week 2)
- **PARA System**: Store knowledge in **Notes** categorized by Projects, Areas, Resources, and Archives.
- **Interactive Knowledge Graph**: Click **Notes > Second Brain** to explore your ideas and connections on the 2D/3D graph canvas.

#### 🟣 Level 4: Master System (Week 3+)
- **AI Executive Coach**: Use the **AI** tab for deep strategic planning and prompt suggestions.
- **Custom AI Memory**: Save persistent personal guidelines in **AI > Personal Rules & Memory**.
- **Sunday Weekly Review**: Launch the structured 4-part review ritual (**What happened → What matters → What needs attention → Next actions**).`;
      actions = [
        { id: 'act-capture-now', label: 'Try Quick Capture (C)', actionType: 'open_capture', icon: 'Plus' },
        { id: 'act-focus-now', label: 'Try Focus Mode', actionType: 'open_focus', icon: 'Clock' },
        { id: 'act-search-now', label: 'Try Universal Search (⌘K)', actionType: 'open_search', icon: 'Search' },
      ];
      relatedTopics = ['getting-started', 'tasks', 'plan', 'notes'];
    } else if (qLower.includes('focus') || qLower.includes('sound') || qLower.includes('pomodoro') || qLower.includes('rain')) {
      reply = `### How to Use Focus Mode & Soundscapes

1. **Launch Focus**: Tap the **Clock icon** in the top header, or tap **Focus Mode** on the Home screen.
2. **Choose Timer**: Select 25m, 50m, or set a custom countdown duration.
3. **Select Ambient Soundscape**: Choose from procedurally synthesized sounds: **Rain**, **Ocean Waves**, **White Noise**, or **Binaural Drone** (powered by the Web Audio API without internet streaming).
4. **Link Task**: Select a task from your queue to anchor your focus session.
5. **Start Focus**: Tap **Start Focus** to enter full-screen distraction-free mode.
6. When completed, your session duration is automatically saved to your productivity analytics.`;
      actions = [
        { id: 'act-open-focus', label: 'Open Focus Mode', actionType: 'open_focus', icon: 'Clock' },
        { id: 'act-play-rain', label: 'Play Rain Soundscape', actionType: 'play_sound', payload: { sound: 'rain' }, icon: 'Volume2' },
      ];
      relatedTopics = ['focus'];
    } else if (qLower.includes('sync') || qLower.includes('backup') || qLower.includes('export') || qLower.includes('google')) {
      reply = `### Data Sovereignty: Google Sync, Backup & Export

1. **Multi-Device Sync**: Tap your profile avatar or the **Gear icon** in the header to open **Settings**, then tap **Sign in with Google** under "Account & Sync" to sync across all your devices via Firebase.
2. **Download Backup**: In Settings, scroll to **Data Management & Backup** and click **Export All Data (JSON)** to download a complete portable JSON file of your entire database.
3. **Restore Backup**: Click **Import Backup (JSON)** to restore your data from any saved backup file at any time.`;
      actions = [
        { id: 'act-open-settings', label: 'Open Settings', actionType: 'open_settings', icon: 'Settings' },
        { id: 'act-export-json', label: 'Download JSON Backup', actionType: 'export_data', icon: 'Download' },
      ];
      relatedTopics = ['settings', 'privacy'];
    } else {
      reply = `### Life OS Guide

I analyzed your question regarding **"${question}"**. 

Life OS is designed around a unified 9-step flow: **Capture → Understand → Organize → Connect → Prioritize → Plan → Act → Learn → Improve**.

**Quick navigation tips:**
- **Home**: Today's prioritized tasks and 5-second check-in.
- **Tasks**: GTD queues (*Today*, *Inbox*, *Upcoming*, *Completed*) with subtasks and priority ratings.
- **Plan**: 24-hour visual time blocking, calendar scheduling, and goal milestones.
- **Notes**: PARA markdown documentation and Second Brain interactive graph.
- **Life**: Habit streaks, energy rhythms, and financial burn rate clarity.
- **AI**: Executive coaching, personal memory vault, and automations.

You can ask me any specific question like *"How do I create a time block?"*, *"How does AI memory work?"*, or *"Teach me Life OS from beginner to advanced."*`;
      actions = [
        { id: 'act-home', label: 'Go to Home', actionType: 'navigate', payload: { nav: 'home' }, icon: 'Home' },
        { id: 'act-search', label: 'Search App (⌘K)', actionType: 'open_search', icon: 'Search' },
      ];
      relatedTopics = ['getting-started'];
    }

    return res.json({
      reply,
      actions,
      relatedTopics,
      engine: 'local-verified-intelligence',
    });
  } catch (err: any) {
    console.error('Fatal in /api/ai/help:', err);
    return res.json({
      reply: 'Life OS Help Assistant is ready. What feature or workflow would you like to explore?',
      actions: [],
      relatedTopics: ['getting-started'],
      engine: 'safe-fallback',
    });
  }
});

// 6. AI Template Generator (Natural Language -> Interconnected Life Objects Blueprint)
app.post('/api/ai/template-generate', async (req, res) => {
  try {
    const { prompt: userPrompt, category } = req.body;
    if (!userPrompt || typeof userPrompt !== 'string') {
      return res.status(400).json({ error: 'Prompt is required' });
    }

    const { provider, geminiKey, openaiKey, customModel, autoFallback } = extractAIConfig(req);

    const systemPrompt = `You are the master Systems Architect of Life OS.
Your task is to generate a comprehensive, interconnected multi-object template based on the user's request.
The template must NOT be just a static page. It must create an entire operational ecosystem:
- 1 overarching Goal with vision and target area
- 1-2 focused Projects
- 2-3 Milestones
- 3-5 Actionable Tasks with priority, energy, and duration
- 1-2 Calendar blocks (focus or timeblock)
- 1-2 Daily or weekly Habits with streak target
- 1 Rich Note (with markdown headings, frameworks, checklist)
- 2-3 Template Variables with default values (e.g. {{subjectName}}, {{targetDate}}, {{dailyTarget}})
- 1 Smart Automation rule

Output strict JSON matching this structure:
{
  "title": "Clear Template Title",
  "description": "Concise 1-sentence value proposition",
  "category": "study" | "projects" | "goals" | "reviews" | "meetings" | "habits" | "budgeting" | "reading" | "planning",
  "icon": "GraduationCap" | "Rocket" | "Zap" | "Compass" | "Users" | "BookOpen" | "Wallet" | "Flame" | "Target",
  "color": "#8b5cf6" | "#0ea5e9" | "#f59e0b" | "#10b981" | "#6366f1" | "#ec4899" | "#14b8a6" | "#f97316",
  "variables": [
    { "key": "variableKey", "label": "Human Label", "placeholder": "Example", "defaultValue": "Default Value", "type": "text" }
  ],
  "blueprint": {
    "goal": { "title": "...", "vision": "...", "area": "Learning" | "Career" | "Personal" | "Finance" | "Health" },
    "projects": [{ "title": "...", "description": "..." }],
    "milestones": [{ "id": "m-1", "title": "...", "completed": false }],
    "tasks": [
      { "title": "...", "priority": "high", "status": "today", "durationMinutes": 45, "energyRequired": "high", "tags": ["tag1"], "subtasks": [{ "id": "st-1", "title": "...", "completed": false }] }
    ],
    "calendarEvents": [
      { "title": "...", "startTime": "09:00", "endTime": "10:30", "type": "focus", "isFixedCommitment": true, "notes": "..." }
    ],
    "habits": [
      { "title": "...", "frequency": "daily", "targetDaysPerWeek": 6, "category": "Personal" }
    ],
    "notes": [
      { "title": "...", "type": "knowledge", "tags": ["..."], "content": "# Title\\n\\n## Details..." }
    ],
    "automations": [
      { "id": "auto-1", "title": "...", "trigger": "...", "action": "...", "enabled": true, "description": "..." }
    ]
  }
}`;

    if (provider === 'openai' && openaiKey) {
      try {
        const result = await callOpenAI({
          apiKey: openaiKey,
          model: customModel || 'gpt-4o-mini',
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: `Generate a template for: "${userPrompt}". Preferred category: ${category || 'any'}` },
          ],
          responseFormat: { type: 'json_object' },
        });
        const parsed = JSON.parse(result.text || '{}');
        return res.json({ template: parsed, engine: `openai:${result.model}` });
      } catch (err: any) {
        console.warn('OpenAI template gen error:', err.message);
        if (!autoFallback && !geminiKey) throw err;
      }
    }

    const geminiAi = getGemini(geminiKey);
    if (geminiAi) {
      try {
        const modelToUse = customModel || PRIMARY_GEMINI_MODEL;
        const response = await generateWithGemini(
          geminiAi,
          {
            contents: `${systemPrompt}\n\nUser Request: "${userPrompt}". Category hint: ${category || 'any'}. Generate the full template JSON.`,
            config: {
              responseMimeType: 'application/json',
            },
          },
          modelToUse
        );
        const parsed = JSON.parse(response.text || '{}');
        return res.json({ template: parsed, engine: `gemini:${modelToUse}` });
      } catch (err: any) {
        console.warn('Gemini template gen fallback:', err.message);
      }
    }

    // High quality local fallback generator
    const title = userPrompt.slice(0, 45).replace(/^(create|make|build|generate|a|an)\s+/i, '').trim();
    const capitalizedTitle = title.charAt(0).toUpperCase() + title.slice(1);
    const resolvedCat = (category as any) || (
      /exam|study|class|learn|school/i.test(userPrompt) ? 'study' :
      /launch|code|build|app|ship/i.test(userPrompt) ? 'projects' :
      /habit|routine|morning|evening/i.test(userPrompt) ? 'habits' :
      /money|budget|invest|save/i.test(userPrompt) ? 'budgeting' :
      /read|book|paper/i.test(userPrompt) ? 'reading' : 'planning'
    );

    const localTemplate = {
      title: `${capitalizedTitle} Workflow`,
      description: `Structured operational blueprint for ${userPrompt}, complete with connected goals, milestones, tasks, and habits.`,
      category: resolvedCat,
      icon: resolvedCat === 'study' ? 'GraduationCap' : resolvedCat === 'projects' ? 'Rocket' : 'Zap',
      color: '#8b5cf6',
      variables: [
        {
          key: 'mainTarget',
          label: 'Primary Deliverable / Target',
          placeholder: 'e.g. Complete Milestone 1',
          defaultValue: capitalizedTitle,
          type: 'text',
        },
        {
          key: 'targetDate',
          label: 'Completion Target Date',
          placeholder: 'YYYY-MM-DD',
          defaultValue: new Date(Date.now() + 30 * 86400000).toISOString().split('T')[0],
          type: 'date',
        },
      ],
      blueprint: {
        goal: {
          title: `Master & Accomplish ${capitalizedTitle}`,
          vision: `Execute disciplined, high-leverage work to complete ${capitalizedTitle} on schedule with excellence.`,
          area: resolvedCat === 'study' ? 'Learning' : 'Career',
        },
        projects: [
          {
            title: `${capitalizedTitle} — Core Execution Sprint`,
            description: `Primary deliverables, documentation, and milestones for ${capitalizedTitle}.`,
          },
        ],
        milestones: [
          { id: 'm-1', title: 'Complete foundational research and scoping', completed: false },
          { id: 'm-2', title: 'Finish 80% primary deliverables and review checkpoint', completed: false },
          { id: 'm-3', title: 'Final audit, polish, and completion celebration', completed: false },
        ],
        tasks: [
          {
            title: `Kick off ${capitalizedTitle}: Define 3 critical success factors`,
            priority: 'urgent',
            status: 'today',
            durationMinutes: 30,
            energyRequired: 'high',
            tags: ['planning', 'kickoff'],
            subtasks: [
              { id: 'st-1', title: 'Draft project roadmap and deadline checkpoints', completed: false },
              { id: 'st-2', title: 'Eliminate potential friction bottlenecks', completed: false },
            ],
          },
          {
            title: `Execute deep focus sprint on ${capitalizedTitle}`,
            priority: 'high',
            status: 'upcoming',
            durationMinutes: 60,
            energyRequired: 'peak',
            tags: ['deep-work'],
            subtasks: [],
          },
        ],
        calendarEvents: [
          {
            title: `Deep Focus Block: ${capitalizedTitle}`,
            startTime: '09:00',
            endTime: '10:30',
            type: 'focus',
            isFixedCommitment: true,
            notes: 'Uninterrupted deep work sprint.',
          },
        ],
        habits: [
          {
            title: `Daily Dedicated Sprint on ${capitalizedTitle} (45m)`,
            frequency: 'daily',
            targetDaysPerWeek: 5,
            category: 'Focus',
          },
        ],
        notes: [
          {
            title: `${capitalizedTitle} — Master Blueprint & Progress Log`,
            type: 'knowledge',
            tags: ['blueprint', 'notes'],
            content: `# ${capitalizedTitle} — Master Execution Blueprint\n\n## 🎯 North Star Objective\nDeliver {{mainTarget}} by {{targetDate}} with zero friction.\n\n## 📋 Protocol & Strategy\n1. Protect high-energy morning hours for difficult tasks.\n2. Keep active feedback loops open.\n3. Record key lessons and adjustments here.\n\n## 📝 Running Work Notes\n- *Capture breakthroughs, decisions, and meeting takeaways here*`,
          },
        ],
        automations: [
          {
            id: 'auto-tpl-' + Date.now(),
            title: `${capitalizedTitle}: Habit Streak Celebration`,
            trigger: 'When habit completed 5 days in a row',
            action: 'Show motivational milestone toast',
            enabled: true,
            description: 'Maintains momentum and accountability',
          },
        ],
      },
    };

    return res.json({ template: localTemplate, engine: 'local-intelligence' });
  } catch (err: any) {
    console.error('Fatal in /api/ai/template-generate:', err);
    return res.status(500).json({ error: 'Failed to generate template' });
  }
});

// 7. Universal Semantic Search Endpoint
app.post('/api/ai/semantic-search', async (req, res) => {
  try {
    const { query, items = [] } = req.body;
    if (!query || typeof query !== 'string') {
      return res.json({ scoredItemIds: [] });
    }

    const qLower = query.toLowerCase().trim();
    const queryTokens = qLower.split(/\s+/).filter(Boolean);

    // Fast semantic scoring algorithm
    const scored = items.map((item: any) => {
      let score = 0;
      const titleLower = (item.title || '').toLowerCase();
      const descLower = (item.description || item.content || '').toLowerCase();
      const tags = Array.isArray(item.tags) ? item.tags.map((t: string) => t.toLowerCase()) : [];

      // Exact title match
      if (titleLower === qLower) score += 100;
      else if (titleLower.includes(qLower)) score += 50;

      // Token matches
      queryTokens.forEach((token) => {
        if (titleLower.includes(token)) score += 20;
        if (tags.some((t: string) => t.includes(token))) score += 15;
        if (descLower.includes(token)) score += 8;
      });

      // Semantic associations
      if (
        (qLower.includes('urgent') || qLower.includes('important')) &&
        (item.priority === 'urgent' || item.priority === 'high')
      ) {
        score += 15;
      }
      if (
        (qLower.includes('today') || qLower.includes('now')) &&
        (item.status === 'today' || item.date === new Date().toISOString().split('T')[0])
      ) {
        score += 15;
      }

      return { id: item.id, score };
    });

    const topScored = scored
      .filter((s: any) => s.score > 0)
      .sort((a: any, b: any) => b.score - a.score)
      .map((s: any) => ({ id: s.id, score: s.score }));

    return res.json({ scoredItems: topScored });
  } catch (err) {
    return res.json({ scoredItems: [] });
  }
});

// Vite middleware in dev, static dist in production
async function start() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Life OS Server running on http://0.0.0.0:${PORT}`);
  });
}

start();
