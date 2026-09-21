#!/usr/bin/env bash
# Why does web_search return "no search results were returned" for well-formed
# queries? The stored tool arguments ruled out bad queries -- "Austin High
# School Texas football" is a perfectly good search -- so the failure is on
# DuckDuckGo's side. This checks whether the User-Agent is what decides it.
#
# tools.js sends 'BrittainWebChat/1.0' to https://html.duckduckgo.com/html/.
probe() {
    local label="$1" agent="$2" url="$3" query="$4"
    local body="/tmp/ddg_$(echo "$label$query" | md5sum | cut -c1-8).html"
    local code size
    read -r code size <<<"$(curl -s -o "$body" -w '%{http_code} %{size_download}' \
        -X POST -A "$agent" -H 'Accept: text/html' \
        --data-urlencode "q=$query" "$url")"
    local hits unsupported
    hits=$(grep -c 'result__a' "$body")
    unsupported=$(grep -c 'browser is not supported' "$body")
    printf '  %-26s %-34s http=%s bytes=%-6s results=%-3s stub=%s\n' \
        "$label" "$query" "$code" "$size" "$hits" "$unsupported"
}

BROWSER='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36'

echo "html endpoint, tools.js User-Agent:"
for q in "Austin High School Texas football" "Austin High School Texas" "NFL players list"; do
    probe "BrittainWebChat/1.0" "BrittainWebChat/1.0" "https://html.duckduckgo.com/html/" "$q"
done

echo
echo "html endpoint, a browser User-Agent:"
for q in "Austin High School Texas football" "Austin High School Texas" "NFL players list"; do
    probe "browser UA" "$BROWSER" "https://html.duckduckgo.com/html/" "$q"
done

echo
echo "lite endpoint (what the stub page points at), tools.js User-Agent:"
for q in "Austin High School Texas football" "NFL players list"; do
    probe "lite + BrittainWebChat" "BrittainWebChat/1.0" "https://lite.duckduckgo.com/lite/" "$q"
done
