import { GoogleGenAI, Type } from "@google/genai";
import { AnalysisResult } from "../types";

// Initialize Gemini Client
// IMPORTANT: The API key is injected via process.env.API_KEY
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Analyzes audio using Google's Gemini 3 Pro model with "Thinking" enabled for maximum forensic accuracy.
 * Includes retry logic and fallback to Gemini 3 Flash for reliability.
 */
export const analyzeAudio = async (
  base64Data: string, 
  mimeType: string, 
  language: string
): Promise<AnalysisResult> => {
  
  // Highly specific system prompt for Audio Forensics
  const prompt = `
      You are a specialized Audio Forensics AI participating in a Deepfake Detection Challenge.
      
      TARGET: Classify the input audio as either 'AI_GENERATED' or 'HUMAN'.
      LANGUAGE: ${language}
      
      EVALUATION CRITERIA:
      1. **Breath & Pauses**: Real humans breathe. AI often forgets to breathe or places breaths at unnatural intervals.
      2. **Prosody & Intonation**: Human speech has irregular pitch curves. AI often produces "flat" or "perfectly cyclic" pitch patterns.
      3. **Spectral Artifacts**: Listen for metallic ringing, phasing, or high-frequency buzz typical of neural vocoders.
      4. **Micro-details**: Lip smacks, tongue clicks, and throat clearing are strong indicators of HUMAN speech.
      5. **Background**: Absolute digital silence between words is a strong indicator of AI_GENERATED.
      
      DECISION LOGIC:
      - If it sounds "too perfect", it is likely AI.
      - If it has natural imperfections (breaths, clicks, variable pace), it is likely HUMAN.
      
      OUTPUT: Return a JSON object with the classification, a confidence score (0.0-1.0), and a brief technical explanation.
    `;

  const requestBody = {
    contents: {
      parts: [
        {
          inlineData: {
            mimeType: mimeType,
            data: base64Data
          }
        },
        {
          text: prompt
        }
      ]
    },
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          classification: {
            type: Type.STRING,
            enum: ["AI_GENERATED", "HUMAN"],
            description: "The definitive classification."
          },
          confidence: {
            type: Type.NUMBER,
            description: "Confidence score (0.0 to 1.0)."
          },
          explanation: {
            type: Type.STRING,
            description: "Technical forensic explanation."
          }
        },
        required: ["classification", "confidence", "explanation"]
      }
    }
  };

  // Helper function to handle API calls with retries and model fallback
  const attemptAnalysis = async (model: string, budget: number, retryCount = 0): Promise<AnalysisResult> => {
    try {
      // Configure thinking budget for reasoning models (Gemini 3 series)
      const configWithThinking = {
        ...requestBody.config,
        thinkingConfig: { thinkingBudget: budget },
        maxOutputTokens: budget + 4096, // Ensure enough tokens for thinking + output
      };

      const response = await ai.models.generateContent({
        model,
        contents: requestBody.contents,
        config: configWithThinking
      });

      const text = response.text;
      if (!text) throw new Error("Empty response from Gemini");
      
      return JSON.parse(text) as AnalysisResult;

    } catch (error: any) {
      const msg = error?.message || '';
      const isQuota = msg.includes('429') || msg.includes('quota') || msg.includes('resource exhausted');
      
      if (isQuota) {
        console.warn(`[Quota Limit] Model: ${model}, Attempt: ${retryCount + 1}`);
        
        // 1. Retry with backoff (up to 2 times)
        if (retryCount < 2) {
          await wait(2000 * (retryCount + 1));
          return attemptAnalysis(model, budget, retryCount + 1);
        }
        
        // 2. Fallback to Flash if Pro fails repeatedly
        if (model === 'gemini-3-pro-preview') {
          console.log("Switching to Gemini 3 Flash for fallback...");
          return attemptAnalysis('gemini-3-flash-preview', 1024, 0);
        }
      }
      
      throw error;
    }
  };

  // Start with the most powerful model
  return attemptAnalysis('gemini-3-pro-preview', 2048);
};