document.addEventListener("DOMContentLoaded", () => {
  const blocks = document.querySelectorAll("div.highlight");

  for (const block of blocks) {
    if (block.querySelector(".ff-copy-button")) {
      continue;
    }

    const pre = block.querySelector("pre");
    if (!pre) {
      continue;
    }

    const code = pre.querySelector("code");
    const text = (code ? code.textContent : pre.textContent) || "";
    if (!text.trim()) {
      continue;
    }

    const button = document.createElement("button");
    button.type = "button";
    button.className = "ff-copy-button";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy code block");
    const toolbar = document.createElement("div");
    toolbar.className = "ff-copy-toolbar";
    toolbar.appendChild(button);

    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(text);
      } catch (error) {
        const area = document.createElement("textarea");
        area.value = text;
        area.style.position = "fixed";
        area.style.opacity = "0";
        document.body.appendChild(area);
        area.focus();
        area.select();
        document.execCommand("copy");
        document.body.removeChild(area);
      }

      const oldLabel = button.textContent;
      button.textContent = "Copied";
      button.classList.add("ff-copy-button--done");
      window.setTimeout(() => {
        button.textContent = oldLabel;
        button.classList.remove("ff-copy-button--done");
      }, 1200);
    });

    block.insertBefore(toolbar, pre);
    block.classList.add("ff-copyable-block");
  }
});
