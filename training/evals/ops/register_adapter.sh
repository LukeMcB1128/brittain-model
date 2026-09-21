#!/usr/bin/env bash
# Wait for the server, then register run 2's final checkpoint.
#
# The name carries the run, because run 1 also has a step-0100 and the two are
# different models. A transcript log or a score that cannot say which run it
# measured is worth very little.
KEY=$(cat "$HOME/.brittain4_key")
for _ in $(seq 1 40); do
    code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' http://localhost:11435/health)
    if [ "$code" = "200" ]; then
        echo "server up"
        break
    fi
    sleep 8
done

code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' http://localhost:11435/health)
if [ "$code" != "200" ]; then
    echo "server did not come up (health $code)"
    exit 1
fi

register() {
    name="$1"
    path="$2"
    printf '%-22s ' "$name"
    curl -s -X POST http://localhost:11435/v1/load_lora_adapter \
        -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
        -d "{\"lora_name\": \"$name\", \"lora_path\": \"$path\"}"
    echo
}

register run2-step-0116 /home/lukeb/brittain4/adapters/run2/step-0116-mm
# Run 1's best stays loaded so the two can be compared directly, and so the web
# app keeps working -- its gateway asks for this name by default.
register step-0100-mm /home/lukeb/brittain4/adapters/run1/step-0100-mm

echo
echo "registered:"
curl -s http://localhost:11435/v1/models -H "Authorization: Bearer $KEY" \
    | python3 -c 'import json,sys; [print("  " + m["id"]) for m in json.load(sys.stdin)["data"]]'
