import { ReadableHttpKvStore } from "#src/kvstore/http/common.js";
import { joinBaseUrlAndPath } from "#src/kvstore/url.js";
import { StatusMessage } from "#src/status.js";
import { RefCounted } from "#src/util/disposable.js";
import { bigintToStringJsonReplacer } from "#src/util/json.js";
import type { Viewer } from "#src/viewer.js";
import { makeIcon } from "#src/widget/icon.js";

type StateServer = {
  url: string;
  default?: boolean;
};

type StateServers = {
  [name: string]: StateServer;
};

declare const STATE_SERVERS: StateServers | undefined;

export const stateShareEnabled =
  typeof STATE_SERVERS !== "undefined" && Object.keys(STATE_SERVERS).length > 0;

/**
 * Put the share link where the user can get at it.
 *
 * The clipboard is tried first, but it is only AVAILABLE in a secure
 * context: https, or http on localhost/127.0.0.1. An ordinary http origin --
 * an internal hostname on a private address, say -- has no
 * navigator.clipboard at all, so the write throws.
 *
 * This used to live inside the POST's promise chain, under a single catch
 * that reported "Could not access state server." The state had in fact been
 * saved; only the copy failed, and the link was then lost with a message
 * blaming the wrong thing. Testing on localhost could not reveal it, because
 * localhost is one of the origins the browser treats as secure.
 *
 * So: copy when we can, and otherwise show the link in a selectable field
 * and say why. The state exists either way -- that is the part that matters.
 */
async function copyOrShowLink(link: string) {
  try {
    await navigator.clipboard.writeText(link);
    StatusMessage.showTemporaryMessage("Share link copied to clipboard");
    return;
  } catch {
    // fall through to the manual path
  }
  const msg = StatusMessage.showMessage("");
  const text = document.createElement("div");
  text.textContent = window.isSecureContext
    ? "State saved. Could not reach the clipboard -- copy the link:"
    : "State saved. The clipboard needs https (or localhost), so copy the " +
      "link by hand:";
  const field = document.createElement("input");
  field.type = "text";
  field.readOnly = true;
  field.value = link;
  field.style.width = "100%";
  field.style.marginTop = "4px";
  const close = document.createElement("button");
  close.textContent = "Close";
  close.style.marginTop = "4px";
  close.addEventListener("click", () => msg.dispose());
  msg.element.appendChild(text);
  msg.element.appendChild(field);
  msg.element.appendChild(close);
  // Selected and focused, so one ctrl-C is enough. Not auto-dismissed: a
  // timeout here would take the link away mid-copy.
  field.focus();
  field.select();
}

export class StateShare extends RefCounted {
  // call it a widget? no because it doesn't pop out?
  element = document.createElement("div");
  button = makeIcon({ text: "Share", title: "Share State" });
  selectStateServerElement?: HTMLSelectElement;

  constructor(viewer: Viewer) {
    super();

    if (typeof STATE_SERVERS === "undefined") {
      throw new Error(
        "Cannot construct StateSare without defining STATE_SERVERS",
      );
    }

    // if more than one state server, add UI so users can select the state server to use
    if (Object.keys(STATE_SERVERS).length > 1) {
      const selectEl = document.createElement("select");
      selectEl.style.marginRight = "5px";

      this.registerDisposer(
        viewer.selectedStateServer.changed.add(() => {
          const valueFromState = viewer.selectedStateServer.value;
          if (
            Object.values(STATE_SERVERS)
              .map((s) => s.url)
              .includes(valueFromState)
          ) {
            selectEl.value = valueFromState;
          }
        }),
      );

      this.registerEventListener(selectEl, "change", () => {
        viewer.selectedStateServer.value = selectEl.value;
      });

      for (const [name, stateServer] of Object.entries(STATE_SERVERS)) {
        const option = document.createElement("option");
        option.textContent = name;
        option.value = stateServer.url;
        option.selected = !!stateServer.default;
        selectEl.appendChild(option);
      }

      this.element.appendChild(selectEl);
      this.selectStateServerElement = selectEl;
    }

    this.element.appendChild(this.button);

    this.registerEventListener(this.button, "click", () => {
      const selectedStateServer = this.selectStateServerElement
        ? this.selectStateServerElement.value
        : Object.values(STATE_SERVERS)[0].url;

      const { store, path } =
        viewer.dataSourceProvider.sharedKvStoreContext.kvStoreContext.getKvStore(
          selectedStateServer,
        );

      if (!(store instanceof ReadableHttpKvStore)) {
        throw new Error(
          `Non-HTTP protocol not supported: ${selectedStateServer}`,
        );
      }

      StatusMessage.forPromise(
        store
          .fetchOkImpl(joinBaseUrlAndPath(store.baseUrl, path), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(
              viewer.state.toJSON(),
              bigintToStringJsonReplacer,
            ),
          })
          .then((response) => response.json())
          .then((res) => {
            const stateUrlProtcol = new URL(res).protocol;
            const stateUrlWithoutProtocol = res.substring(
              stateUrlProtcol.length,
            );
            const protocol = new URL(selectedStateServer).protocol;
            // origin + PATHNAME, not origin alone.
            //
            // window.location.origin is scheme://host and deliberately drops
            // the path, which is right only when Neuroglancer is served at
            // the site root. Served from a sub-path -- radagast hosts this at
            // /twigcapture/ -- the shared link pointed at the site root and
            // landed the reader on an unrelated page. pathname keeps it,
            // collapses to "/" at the root, and so is correct either way.
            const base = window.location.origin + window.location.pathname;
            const link = `${base}${base.endsWith("/") ? "" : "/"}#!${protocol}${stateUrlWithoutProtocol}`;
            void copyOrShowLink(link);
          })
          .catch((e) => {
            // Now genuinely about the POST: the clipboard is handled above
            // and no longer reports itself as a server failure.
            StatusMessage.showTemporaryMessage(
              `Could not post the state to ${selectedStateServer}: ${e}`,
              6000,
            );
          }),
        {
          initialMessage: `Posting state to ${selectedStateServer}.`,
          delay: true,
          errorPrefix: "",
        },
      );
    });
  }

  disposed() {
    this.element.remove();
    super.disposed();
  }
}
