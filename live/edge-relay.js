/**
 * Cloudflare Worker — Edge Relay for Groq API
 * Deployed globally. Routes browser requests to Groq with CORS headers.
 * Latency from Thailand: ~30ms (BKK/SG edge) vs 200ms (Germany VPS)
 */
export default {
  async fetch(request, env, ctx) {
    // CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization',
          'Access-Control-Max-Age': '86400',
        }
      });
    }

    // Only allow POST to /v1/chat/completions and /v1/audio/transcriptions
    const url = new URL(request.url);
    const groqPath = url.pathname.replace('/groq', '');
    
    if (!['/v1/chat/completions', '/v1/audio/transcriptions'].includes(groqPath)) {
      return new Response(JSON.stringify({ error: 'Invalid endpoint' }), {
        status: 404,
        headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' }
      });
    }

    // Forward to Groq
    const groqUrl = `https://api.groq.com/openai${groqPath}`;
    
    try {
      const groqResponse = await fetch(groqUrl, {
        method: 'POST',
        headers: {
          'Authorization': request.headers.get('Authorization') || `Bearer ${env.GROQ_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: request.body,
      });

      const data = await groqResponse.json();

      return new Response(JSON.stringify(data), {
        status: groqResponse.status,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
          'Cache-Control': 'no-cache',
        }
      });
    } catch (err) {
      return new Response(JSON.stringify({ error: 'Groq API unreachable' }), {
        status: 502,
        headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' }
      });
    }
  }
};