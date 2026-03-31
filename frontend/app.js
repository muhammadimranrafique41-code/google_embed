const STORAGE_KEY = "stylemate-history";
const HISTORY_LIMIT = 10;

document.addEventListener("DOMContentLoaded", () => {
  const chatForm = document.getElementById("chat-form");
  const messageInput = document.getElementById("message-input");
  const imageUpload = document.getElementById("image-upload");
  const chatMessages = document.getElementById("chat-messages");
  const typingIndicator = document.getElementById("typing-indicator");
  const imagePreviewContainer = document.getElementById("image-preview-container");
  const previewImage = document.getElementById("preview-image");
  const removeImageButton = document.getElementById("remove-image");
  const clearButton = document.getElementById("clear-btn");
  const promptButtons = document.querySelectorAll(".prompt-chip");
  const messageTemplate = document.getElementById("message-template");
  const productCardTemplate = document.getElementById("product-card-template");

  let selectedImage = null;
  let history = loadHistory();

  autoResize(messageInput);
  renderHistory();
  renderEmptyStateIfNeeded();

  messageInput.addEventListener("input", () => autoResize(messageInput));

  promptButtons.forEach((button) => {
    button.addEventListener("click", () => {
      messageInput.value = button.dataset.prompt || "";
      autoResize(messageInput);
      messageInput.focus();
    });
  });

  imageUpload.addEventListener("change", (event) => {
    const [file] = event.target.files || [];
    if (!file) {
      clearSelectedImage();
      return;
    }

    selectedImage = file;
    const previewUrl = URL.createObjectURL(file);
    previewImage.src = previewUrl;
    imagePreviewContainer.classList.remove("hidden");
    imagePreviewContainer.setAttribute("aria-hidden", "false");
  });

  removeImageButton.addEventListener("click", () => {
    clearSelectedImage();
  });

  clearButton.addEventListener("click", () => {
    history = [];
    persistHistory();
    chatMessages.innerHTML = "";
    renderEmptyStateIfNeeded();
  });

  chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const message = messageInput.value.trim();
    if (!message && !selectedImage) {
      return;
    }

    removeEmptyState();

    const userText = message || "Find products similar to this image.";
    appendMessage({ role: "user", text: userText });
    pushHistory({ role: "user", text: userText });

    const formData = new FormData();
    formData.append("message", message);
    if (selectedImage) {
      formData.append("image", selectedImage);
    }

    setSubmitting(true);

    try {
      const endpoint = selectedImage ? "/api/chat-image" : "/api/chat";
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Request failed");
      }

      const assistantMessage = {
        role: "assistant",
        text: payload.response || "I found some options for you.",
        products: Array.isArray(payload.products) ? payload.products : [],
      };

      appendMessage(assistantMessage);
      pushHistory(assistantMessage);
    } catch (error) {
      const failureMessage = {
        role: "assistant",
        text: error.message || "Something went wrong. Please try again.",
        products: [],
      };
      appendMessage(failureMessage);
      pushHistory(failureMessage);
    } finally {
      setSubmitting(false);
      messageInput.value = "";
      autoResize(messageInput);
      clearSelectedImage();
      messageInput.focus();
    }
  });

  function appendMessage(entry) {
    const fragment = messageTemplate.content.cloneNode(true);
    const article = fragment.querySelector(".message");
    const textNode = fragment.querySelector(".message-text");

    article.classList.add(entry.role);
    textNode.textContent = entry.text;

    if (entry.role === "assistant" && entry.products?.length) {
      const strip = document.createElement("div");
      strip.className = "products-strip";
      entry.products.forEach((product) => {
        strip.appendChild(buildProductCard(product));
      });
      fragment.querySelector(".message-bubble").appendChild(strip);
    }

    chatMessages.appendChild(fragment);
    scrollToBottom();
  }

  function buildProductCard(product) {
    const fragment = productCardTemplate.content.cloneNode(true);
    const image = fragment.querySelector(".product-image");
    const category = fragment.querySelector(".product-category");
    const name = fragment.querySelector(".product-name");
    const description = fragment.querySelector(".product-description");
    const price = fragment.querySelector(".product-price");
    const rating = fragment.querySelector(".product-rating");
    const link = fragment.querySelector(".product-link");

    image.src = product.image_url || "";
    image.alt = product.product_name || "Product image";
    image.addEventListener("error", () => {
      image.removeAttribute("src");
      image.alt = `${product.product_name || "Product"} image unavailable`;
    });

    category.textContent = product.category || "Catalog item";
    name.textContent = product.product_name || "Untitled product";
    description.textContent = product.description || "No description available.";
    price.textContent = `Rs ${Number(product.price || 0).toFixed(0)}`;
    rating.textContent = `Rating ${Number(product.rating || 0).toFixed(1)}`;
    link.href = product.product_link || "#";

    return fragment;
  }

  function renderHistory() {
    chatMessages.innerHTML = "";
    history.forEach((entry) => appendMessage(entry));
  }

  function renderEmptyStateIfNeeded() {
    if (chatMessages.children.length) {
      return;
    }

    const state = document.createElement("section");
    state.className = "empty-state";
    state.id = "empty-state";
    state.innerHTML = `
      <h3>Start with a style, category, or image</h3>
      <p>Try "show me jeans under Rs 2000" or upload an outfit photo to find visually similar products.</p>
    `;
    chatMessages.appendChild(state);
  }

  function removeEmptyState() {
    const state = document.getElementById("empty-state");
    if (state) {
      state.remove();
    }
  }

  function pushHistory(entry) {
    history.push(entry);
    history = history.slice(-HISTORY_LIMIT);
    persistHistory();
  }

  function loadHistory() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  function persistHistory() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
  }

  function setSubmitting(isSubmitting) {
    typingIndicator.classList.toggle("hidden", !isSubmitting);
    typingIndicator.setAttribute("aria-hidden", String(!isSubmitting));
    messageInput.disabled = isSubmitting;
    imageUpload.disabled = isSubmitting;
    document.getElementById("send-btn").disabled = isSubmitting;
  }

  function clearSelectedImage() {
    if (previewImage.src.startsWith("blob:")) {
      URL.revokeObjectURL(previewImage.src);
    }
    selectedImage = null;
    imageUpload.value = "";
    previewImage.removeAttribute("src");
    imagePreviewContainer.classList.add("hidden");
    imagePreviewContainer.setAttribute("aria-hidden", "true");
  }

  function autoResize(element) {
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 180)}px`;
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
});
