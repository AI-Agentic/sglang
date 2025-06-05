port=31231

curl http://localhost:${port}/start_profile -H "Content-Type: application/json"

sleep 2

curl http://localhost:${port}/stop_profile -H "Content-Type: application/json"