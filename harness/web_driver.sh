#!/bin/bash
# Shell CDP client behind harness/web.json (deps: bash, jq, Chrome).
#
# Harness tool calls are short-lived sh processes, so Chrome runs once as a
# background daemon speaking the DevTools protocol over
# --remote-debugging-pipe: NUL-terminated JSON on fds 3/4, wired to two
# FIFOs under .web/. Holder processes keep the FIFO ends open between
# calls, because Chrome treats a closed pipe as a shutdown request.
# Params arrive as env vars from the harness ($url, $ref, $text, ...).
#
# Chrome starts headless. `show` restarts it with a visible window on the
# same profile and page, so the user can step in on challenge pages and
# logins. The mode sticks (a .web/visible flag file) until `stop`.
#
# A watchdog tears the whole daemon down after IDLE_LIMIT seconds without
# a tool call, so nothing outlives the chat session for long.
#
# Human-facing extras: `status`, `shot` (screenshot to .web/shot.png),
# `stop` (kill the daemon).

set -u

W=.web
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TEXT_CHUNK=3000
IDLE_LIMIT=${WEB_IDLE_LIMIT:-300}

die() { echo "error: $1"; exit 0; }

alive() { [ -f "$W/chrome.pid" ] && kill -0 "$(cat "$W/chrome.pid")" 2>/dev/null; }

# tear the daemon down once tool calls stop coming
start_watchdog() {
    local tick=$(( IDLE_LIMIT / 2 ))
    [ "$tick" -gt 30 ] && tick=30
    [ "$tick" -lt 1 ] && tick=1
    (
        while sleep "$tick"; do
            alive || exit 0
            last=$(stat -f %m "$W/last_used" 2>/dev/null || echo 0)
            [ $(( $(date +%s) - last )) -lt "$IDLE_LIMIT" ] && continue
            bash "$0" stop >/dev/null 2>&1
            exit 0
        done
    ) &
    echo $! > "$W/watchdog.pid"
}

start_chrome() {
    shutdown_chrome
    mkfifo "$W/to" "$W/from"
    local headless="--headless"
    [ -f "$W/visible" ] && headless=""
    "$CHROME" $headless --remote-debugging-pipe --no-first-run \
        --window-size=1280,900 --user-data-dir="$W/profile" about:blank \
        3<"$W/to" 4>"$W/from" 2>>"$W/chrome.log" &
    local cpid=$!
    echo "$cpid" > "$W/chrome.pid"
    # each holder keeps a FIFO end open and exits once Chrome dies
    ( exec 3>"$W/to"; while kill -0 "$cpid" 2>/dev/null; do sleep 5; done ) &
    echo $! > "$W/holders.pid"
    ( exec 3<"$W/from"; while kill -0 "$cpid" 2>/dev/null; do sleep 5; done ) &
    echo $! >> "$W/holders.pid"
    start_watchdog
}

# fd 5 writes to Chrome, fd 6 reads from it
open_fds() { exec 5>"$W/to" 6<"$W/from"; }

# send METHOD [PARAMS_JSON] [SESSION_ID], request id left in SENT_ID
send() {
    SENT_ID=$(( $(cat "$W/seq" 2>/dev/null || echo 100) + 1 ))
    printf '%s' "$SENT_ID" > "$W/seq"
    jq -cn --argjson id "$SENT_ID" --arg m "$1" --argjson p "${2:-null}" --arg s "${3:-}" \
        '{id:$id,method:$m}
         + (if $p != null then {params:$p} else {} end)
         + (if $s != "" then {sessionId:$s} else {} end)' \
        | tr -d '\n' >&5
    printf '\0' >&5
}

# print the reply to request id $1, waiting up to $2 seconds
wait_id() {
    local want=$1 end=$(( SECONDS + $2 )) msg
    while [ "$SECONDS" -lt "$end" ]; do
        IFS= read -r -d '' -t 2 -u 6 msg || continue
        if [ "$(printf '%s' "$msg" | jq -r '.id // empty' 2>/dev/null)" = "$want" ]; then
            printf '%s\n' "$msg"
            return 0
        fi
    done
    return 1
}

# wait up to $1 seconds for a load event, then let the page settle
wait_load() {
    local end=$(( SECONDS + $1 )) msg
    while [ "$SECONDS" -lt "$end" ]; do
        IFS= read -r -d '' -t 2 -u 6 msg || continue
        if [ "$(printf '%s' "$msg" | jq -r '.method // empty' 2>/dev/null)" = "Page.loadEventFired" ]; then
            sleep 1
            return 0
        fi
    done
    return 1
}

attach() {
    send Target.createTarget '{"url":"about:blank"}'
    local t
    t=$(wait_id "$SENT_ID" 20 | jq -r '.result.targetId // empty')
    [ -n "$t" ] || die "chrome did not answer (see $W/chrome.log)"
    send Target.attachToTarget "$(jq -cn --arg t "$t" '{targetId:$t,flatten:true}')"
    SESSION=$(wait_id "$SENT_ID" 10 | jq -r '.result.sessionId // empty')
    [ -n "$SESSION" ] || die "could not attach to the chrome tab"
    printf '%s' "$SESSION" > "$W/session"
    send Page.enable '{}' "$SESSION"
    wait_id "$SENT_ID" 10 >/dev/null
}

ensure() {
    mkdir -p "$W"
    touch "$W/last_used"
    alive || start_chrome
    open_fds
    if [ -s "$W/session" ]; then
        SESSION=$(cat "$W/session")
    else
        attach
    fi
}

# evaluate JS in the page and print its string value
evaluate() {
    send Runtime.evaluate \
        "$(jq -cn --arg e "$1" '{expression:$e,returnByValue:true,awaitPromise:true}')" \
        "$SESSION"
    local r ex
    r=$(wait_id "$SENT_ID" 15) || { echo "error: the page did not answer in time"; return 1; }
    ex=$(printf '%s' "$r" | jq -r '.result.exceptionDetails.exception.description // empty' 2>/dev/null | head -2)
    [ -z "$ex" ] || { echo "js error: $ex"; return 1; }
    printf '%s' "$r" | jq -r '.result.result
        | if has("value") then (.value | if type == "string" then . else tojson end)
          else (.type // "undefined") end'
}

# refresh the one-line page state that feeds the $$page$$ prethink var
update_page() {
    evaluate '(document.title.trim()||"(untitled)")+" :: "+location.href' \
        | head -c 300 > "$W/page.txt"
}

where() { echo "now at: $(cat "$W/page.txt" 2>/dev/null || echo unknown)"; }

# JSON-encode an env var so it embeds safely in a JS expression
jsq() { jq -cn --arg s "$1" '$s'; }

FIND_JS='(()=>{const q=QJ.toLowerCase();let n=0;const out=[];
const sel="a[href],button,input,textarea,select,[role=button],[role=link],[role=tab],[role=searchbox],[onclick]";
for(const e of document.querySelectorAll(sel)){
if(!(e.offsetParent||e.getClientRects().length))continue;
n++;e.setAttribute("data-dref",String(n));
const label=(e.innerText||e.value||e.placeholder||e.getAttribute("aria-label")||e.title||e.name||"").trim().replace(/\s+/g," ").slice(0,70);
const tag=e.tagName.toLowerCase()+(e.type?"/"+e.type:"");
if(q&&!(label+" "+tag).toLowerCase().includes(q))continue;
out.push("["+n+"] "+tag+" "+(label||(e.getAttribute("href")||"").slice(0,80)));
if(out.length>=30){out.push("(more elements not shown, refine with a query)");break}}
return out.join("\n")||"(no interactive elements"+(q?" matching "+QJ:"")+" on this page)"})()'

CLICK_JS='(()=>{const r=RJ;const e=document.querySelector("[data-dref="+JSON.stringify(r)+"]");
if(!e)return "error: no element with ref "+r+" (refs reset on navigation, call find again)";
const label=(e.innerText||e.value||e.getAttribute("aria-label")||e.tagName).trim().replace(/\s+/g," ").slice(0,60);
e.scrollIntoView({block:"center"});e.click();
return "clicked ["+r+"] "+label})()'

TYPE_JS='(()=>{const r=RJ,t=TJ,s=SJ;const e=document.querySelector("[data-dref="+JSON.stringify(r)+"]");
if(!e)return "error: no element with ref "+r+" (refs reset on navigation, call find again)";
e.focus();
const d=Object.getOwnPropertyDescriptor(Object.getPrototypeOf(e),"value");
if(d&&d.set)d.set.call(e,t);else e.value=t;
e.dispatchEvent(new Event("input",{bubbles:true}));
e.dispatchEvent(new Event("change",{bubbles:true}));
if(s){const k={key:"Enter",code:"Enter",keyCode:13,which:13,bubbles:true,cancelable:true};
e.dispatchEvent(new KeyboardEvent("keydown",k));e.dispatchEvent(new KeyboardEvent("keyup",k));
if(e.form){if(e.form.requestSubmit)e.form.requestSubmit();else e.form.submit()}}
return "typed into ["+r+"]"+(s?", submitted":"")})()'

READ_JS='(()=>{const o=OJ;const t=document.body?document.body.innerText:"";
if(!t)return "(the page has no text)";
const chunk=t.slice(o,o+NJ).trim();
if(!chunk)return "(offset "+o+" is past the end, the page text is "+t.length+" chars)";
return chunk+(t.length>o+NJ?"\n\n[chars "+o+".."+(o+NJ)+" of "+t.length+", call read_page with offset="+(o+NJ)+" for more]":"")})()'

cmd_navigate() {
    ensure
    local u=${url:?} err
    case "$u" in http://*|https://*) ;; *) u="https://$u" ;; esac
    send Page.navigate "$(jq -cn --arg u "$u" '{url:$u}')" "$SESSION"
    err=$(wait_id "$SENT_ID" 15 | jq -r '.result.errorText // empty')
    [ -z "$err" ] || die "navigation failed: $err"
    wait_load 20 || true
    update_page
    where
    echo
    evaluate 'document.body?document.body.innerText.slice(0,700).trim():"(no text yet)"'
}

cmd_read() {
    ensure
    local o
    o=$(printf '%s' "${offset:-0}" | tr -cd '0-9')
    local js=${READ_JS//OJ/${o:-0}}
    evaluate "${js//NJ/$TEXT_CHUNK}"
}

cmd_find() {
    ensure
    evaluate "${FIND_JS//QJ/$(jsq "${query:-}")}"
}

cmd_click() {
    ensure
    evaluate "${CLICK_JS//RJ/$(jsq "${ref:?}")}"
    wait_load 5 || true
    update_page
    where
}

cmd_type() {
    ensure
    local s=false
    [ "${submit:-}" = true ] && s=true
    local js=${TYPE_JS//RJ/$(jsq "${ref:?}")}
    js=${js//TJ/$(jsq "${text:?}")}
    evaluate "${js//SJ/$s}"
    if [ "$s" = true ]; then
        wait_load 5 || true
        update_page
        where
    fi
}

cmd_eval() {
    ensure
    evaluate "${js:?}" | head -c 3300
}

cmd_shot() {
    ensure
    send Page.captureScreenshot '{}' "$SESSION"
    wait_id "$SENT_ID" 20 | jq -r '.result.data // empty' | base64 -d > "$W/shot.png"
    echo "wrote $W/shot.png ($(wc -c < "$W/shot.png" | tr -d ' ') bytes)"
}

cmd_status() {
    if alive; then
        echo "chrome running (pid $(cat "$W/chrome.pid"))"
        where
    else
        echo "chrome not running"
    fi
}

shutdown_chrome() {
    [ -f "$W/chrome.pid" ] && kill "$(cat "$W/chrome.pid")" 2>/dev/null
    [ -f "$W/holders.pid" ] && kill $(cat "$W/holders.pid") 2>/dev/null
    [ -f "$W/watchdog.pid" ] && kill "$(cat "$W/watchdog.pid")" 2>/dev/null
    rm -f "$W/chrome.pid" "$W/holders.pid" "$W/watchdog.pid" "$W/last_used" \
        "$W/session" "$W/seq" "$W/to" "$W/from"
}

# restart with a visible window, back on the page we were on
cmd_show() {
    mkdir -p "$W"
    if [ -f "$W/visible" ] && alive; then
        echo "the browser window is already visible"
        where
        return
    fi
    local u=""
    if alive; then
        ensure
        u=$(evaluate "location.href") || u=""
    fi
    touch "$W/visible"
    shutdown_chrome
    ensure
    echo "the browser window is now visible to the user"
    case "$u" in
        http*) url="$u" cmd_navigate ;;
        *) echo "no page loaded yet" ;;
    esac
}

cmd_stop() {
    shutdown_chrome
    rm -f "$W/page.txt" "$W/visible"
    echo "stopped"
}

case "${1:-}" in
    navigate) cmd_navigate ;;
    read)     cmd_read ;;
    find)     cmd_find ;;
    click)    cmd_click ;;
    type)     cmd_type ;;
    eval)     cmd_eval ;;
    show)     cmd_show ;;
    shot)     cmd_shot ;;
    status)   cmd_status ;;
    stop)     cmd_stop ;;
    *) die "usage: web_driver.sh navigate|read|find|click|type|eval|shot|status|stop" ;;
esac
