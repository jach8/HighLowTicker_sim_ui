const WebSocket = require('ws');
const { spawn } = require('child_process');
const http = require('http');
const readline = require('readline');

const server = http.createServer();
const wss = new WebSocket.Server({ server });

wss.on('connection', (ws) => {
    console.log('New WebSocket client connected');

    // Forward messages from the frontend to the Python process
    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message.toString());
            if (data.type === 'setSpeed' || data.type === 'setVolatility') {
                pythonProcess.stdin.write(JSON.stringify(data) + '\n');
            } else if (message.toString() === 'ping') {
                // Ignore ping messages
                return;
            }
        } catch (err) {
            console.error('Error processing WebSocket message:', err);
        }
    });

    ws.on('close', () => {
        console.log('WebSocket client disconnected');
    });
});

const pythonProcess = spawn('python3', ['main.py'], {
    stdio: ['pipe', 'pipe', 'pipe'] // Ensure stdin is a pipe
});

const rl = readline.createInterface({
    input: pythonProcess.stdout,
    terminal: false
});

rl.on('line', (line) => {
    try {
        const stockData = JSON.parse(line.trim());
        wss.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(JSON.stringify(stockData));
            } else {
                console.log('Client not ready, state:', client.readyState);
            }
        });
    } catch (err) {
        // If it's not JSON, assume it's a log message
        console.log('Python log:', line);
    }
});

pythonProcess.stderr.on('data', (data) => {
    console.error('Python error:', data.toString());
});

pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
});

server.listen(8080, () => {
    console.log('WebSocket server running on ws://localhost:8080');
});