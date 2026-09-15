/* Themed, keyboard-accessible single-selects; original controls retain form values. */
(() => {
  "use strict";
  const controls = new Map();
  let opened = null;
  let sequence = 0;
  function enhance(select) {
    if (controls.has(select) || select.multiple || select.size > 1) return;
    const wrapper = document.createElement("div");
    wrapper.className = "themed-select";
    select.before(wrapper);
    wrapper.append(select);
    const trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = "select-trigger";
    trigger.id = `select-trigger-${++sequence}`;
    trigger.setAttribute("role", "combobox");
    trigger.setAttribute("aria-haspopup", "listbox");
    trigger.setAttribute("aria-expanded", "false");
    const text = document.createElement("span");
    text.className = "select-value";
    trigger.append(text);
    const menu = document.createElement("div");
    menu.className = "select-menu";
    menu.id = `select-menu-${sequence}`;
    menu.setAttribute("role", "listbox");
    menu.hidden = true;
    trigger.setAttribute("aria-controls", menu.id);
    const labels = [...select.labels];
    labels.forEach((label, i) => {
      if (!label.id) label.id = `select-label-${sequence}-${i}`;
      label.htmlFor = trigger.id;
    });
    if (labels.length) {
      const ids = labels.map((label) => label.id).join(" ");
      trigger.setAttribute("aria-labelledby", ids);
      menu.setAttribute("aria-labelledby", ids);
    } else {
      trigger.setAttribute(
        "aria-label",
        select.getAttribute("aria-label") || "Choose an option",
      );
    }
    wrapper.append(trigger, menu);
    select.hidden = true;
    let active = -1;
    let search = "";
    let searchTime = 0;
    function close() {
      menu.hidden = true;
      trigger.setAttribute("aria-expanded", "false");
      trigger.removeAttribute("aria-activedescendant");
      wrapper.classList.remove("is-open", "opens-up");
      if (opened === state) opened = null;
    }
    function mark() {
      [...menu.children].forEach((item, i) =>
        item.classList.toggle("is-active", i === active),
      );
      const item = menu.children[active];
      if (item && !menu.hidden) {
        trigger.setAttribute("aria-activedescendant", item.id);
        item.scrollIntoView?.({ block: "nearest" });
      }
    }
    function sync() {
      text.textContent =
        select.options[select.selectedIndex]?.textContent || "Choose an option";
      trigger.disabled = select.disabled;
      trigger.setAttribute("aria-required", String(select.required));
      menu.replaceChildren();
      [...select.options].forEach((option, index) => {
        const item = document.createElement("div");
        item.className = "select-option";
        item.id = `${menu.id}-option-${index}`;
        item.setAttribute("role", "option");
        item.setAttribute(
          "aria-selected",
          String(index === select.selectedIndex),
        );
        item.setAttribute(
          "aria-disabled",
          String(option.disabled || option.parentElement.disabled === true),
        );
        item.textContent = option.textContent;
        item.addEventListener("pointerdown", (event) => event.preventDefault());
        item.addEventListener("click", () => commit(index));
        menu.append(item);
      });
      active = select.selectedIndex;
      if (select.disabled) close();
      mark();
    }
    function enabled(index) {
      const option = select.options[index];
      return (
        option && !option.disabled && option.parentElement.disabled !== true
      );
    }
    function open() {
      if (trigger.disabled) return;
      if (opened && opened !== state) opened.close();
      sync();
      menu.hidden = false;
      wrapper.classList.add("is-open");
      const rect = trigger.getBoundingClientRect();
      wrapper.classList.toggle(
        "opens-up",
        window.innerHeight - rect.bottom < 220 &&
          rect.top > window.innerHeight - rect.bottom,
      );
      trigger.setAttribute("aria-expanded", "true");
      opened = state;
      if (!enabled(active))
        active = [...select.options].findIndex((_, i) => enabled(i));
      mark();
    }
    function commit(index) {
      if (!enabled(index) || select.disabled) return;
      const changed = select.selectedIndex !== index;
      select.selectedIndex = index;
      sync();
      close();
      trigger.focus();
      if (changed) {
        select.dispatchEvent(new Event("input", { bubbles: true }));
        select.dispatchEvent(new Event("change", { bubbles: true }));
      }
    }
    const state = { sync, close, wrapper };
    controls.set(select, state);
    trigger.addEventListener("click", () => (menu.hidden ? open() : close()));
    trigger.addEventListener("keydown", (event) => {
      if (event.key === "Tab") {
        close();
        return;
      }
      if (event.key === "Escape") {
        event.preventDefault();
        close();
        return;
      }
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        if (menu.hidden) open();
        else commit(active);
        return;
      }
      if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
        event.preventDefault();
        const wasClosed = menu.hidden;
        if (wasClosed) open();
        const indexes = [...select.options].map((_, i) => i).filter(enabled);
        if (!indexes.length) return;
        if (event.key === "Home") active = indexes[0];
        else if (event.key === "End") active = indexes.at(-1);
        else if (!wasClosed) {
          const position = indexes.indexOf(active);
          active =
            indexes[
              Math.max(
                0,
                Math.min(
                  indexes.length - 1,
                  position + (event.key === "ArrowDown" ? 1 : -1),
                ),
              )
            ];
        }
        mark();
        return;
      }
      if (
        event.key.length === 1 &&
        !event.ctrlKey &&
        !event.metaKey &&
        !event.altKey
      ) {
        event.preventDefault();
        if (menu.hidden) open();
        const time = Date.now();
        search = time - searchTime > 700 ? event.key : search + event.key;
        searchTime = time;
        const repeated = [...search].every(
          (c) => c.toLowerCase() === search[0].toLowerCase(),
        );
        const query = (repeated ? search[0] : search).toLowerCase();
        for (let offset = 1; offset <= select.options.length; offset++) {
          const index =
            (active + offset + select.options.length) % select.options.length;
          if (
            enabled(index) &&
            select.options[index].textContent
              .trim()
              .toLowerCase()
              .startsWith(query)
          ) {
            active = index;
            mark();
            break;
          }
        }
      }
    });
    wrapper.addEventListener("focusout", (event) => {
      if (!wrapper.contains(event.relatedTarget)) close();
    });
    select.addEventListener("change", sync);
    sync();
  }
  const scan = (node) => {
    if (node.nodeType !== 1) return;
    if (node.tagName === "SELECT") enhance(node);
    node.querySelectorAll("select").forEach(enhance);
  };
  scan(document.body);
  new MutationObserver((records) => {
    const dirty = new Set();
    for (const record of records) {
      if (record.type === "childList") record.addedNodes.forEach(scan);
      const select = record.target.closest?.("select");
      if (select && controls.has(select)) dirty.add(select);
    }
    dirty.forEach((select) => controls.get(select).sync());
    for (const [select, state] of controls) {
      if (!select.isConnected) {
        state.close();
        controls.delete(select);
      }
    }
    if (opened?.wrapper.closest("[hidden]")) opened.close();
  }).observe(document.body, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeFilter: ["disabled", "selected", "hidden", "label", "required"],
  });
  document.addEventListener("pointerdown", (event) => {
    if (opened && !opened.wrapper.contains(event.target)) opened.close();
  });
})();
