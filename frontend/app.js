const API_URL = "http://127.0.0.1:8000/chat";

const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("composer");
const inputEl = document.getElementById("question-input");
const sendButtonEl = document.getElementById("send-button");

function showEmptyStateIfNeeded() {
  if (messagesEl.children.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No messages yet — ask a question about the documents on file to get started.";
    messagesEl.appendChild(empty);
  }
}

function clearEmptyState() {
  const empty = messagesEl.querySelector(".empty-state");
  if (empty) empty.remove();
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addUserMessage(text) {
  clearEmptyState();
  const wrapper = document.createElement("div");
  wrapper.className = "message user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  scrollToBottom();
}

function addLoadingMessage() {
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant";
  wrapper.dataset.loading = "true";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = '<span class="loading-dots"><span></span><span></span><span></span></span>';
  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  scrollToBottom();
  return wrapper;
}

function renderAnswer(wrapper, data) {
  const bubble = wrapper.querySelector(".bubble");
  bubble.innerHTML = "";
  bubble.textContent = data.answer;

  if (data.sources && data.sources.length > 0) {
    // Per design: just the filename(s), not the full chunk text.
    const uniqueFilenames = [...new Set(data.sources.map((s) => s.filename))];
    const sourceLine = document.createElement("div");
    sourceLine.className = "source-line";
    sourceLine.textContent =
      uniqueFilenames.length === 1
        ? `Source: ${uniqueFilenames[0]}`
        : `Sources: ${uniqueFilenames.join(", ")}`;
    wrapper.appendChild(sourceLine);
  }
}

function renderNoMatch(wrapper, message) {
  const bubble = wrapper.querySelector(".bubble");
  bubble.classList.add("no-match");
  bubble.textContent = message;
}

function renderError(wrapper, message) {
  const bubble = wrapper.querySelector(".bubble");
  bubble.classList.add("error");
  bubble.textContent = message;
}

async function askQuestion(question) {
  const loadingWrapper = addLoadingMessage();
  delete loadingWrapper.dataset.loading;

  let response;
  try {
    response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch (networkError) {
    renderError(
      loadingWrapper,
      "Couldn't reach the server. Check that the API is running and try again."
    );
    scrollToBottom();
    return;
  }

  if (!response.ok) {
    let detail = `The server returned an error (status ${response.status}).`;
    try {
      const body = await response.json();
      if (body && body.detail) detail = body.detail;
    } catch (_) {
      // response body wasn't JSON — keep the generic message above.
    }
    renderError(loadingWrapper, detail);
    scrollToBottom();
    return;
  }

  let data;
  try {
    data = await response.json();
  } catch (parseError) {
    renderError(loadingWrapper, "Received an unreadable response from the server.");
    scrollToBottom();
    return;
  }

  if (data.type === "answer") {
    renderAnswer(loadingWrapper, data);
  } else if (data.type === "no_match") {
    renderNoMatch(loadingWrapper, data.message);
  } else {
    renderError(loadingWrapper, "Received an unexpected response from the server.");
  }
  scrollToBottom();
}

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = inputEl.value.trim();
  if (!question) return;

  addUserMessage(question);
  inputEl.value = "";
  inputEl.disabled = true;
  sendButtonEl.disabled = true;

  askQuestion(question).finally(() => {
    inputEl.disabled = false;
    sendButtonEl.disabled = false;
    inputEl.focus();
  });
});

showEmptyStateIfNeeded();
