const API_BASE = '/api';

export async function getHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, {
      headers: { 'Accept': 'application/json' },
    });
    if (!res.ok) {
      return { status: 'error', message: `HTTP ${res.status}` };
    }
    const data = await res.json();
    return { status: 'ok', ...data };
  } catch (err) {
    return { status: 'offline', error: err.message };
  }
}

export async function postAnswer(query) {
  const res = await fetch(`${API_BASE}/answer`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify({ query: query.trim() }),
  });

  if (!res.ok) {
    let errorDetail = 'Network response was not ok';
    try {
      const errJson = await res.json();
      if (errJson.detail) errorDetail = errJson.detail;
    } catch (_) {}
    throw new Error(errorDetail);
  }

  return res.json();
}