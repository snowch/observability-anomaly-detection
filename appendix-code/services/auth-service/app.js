const express = require('express');
const app = express();

app.use(express.json());

// Simple auth endpoint for demo purposes
app.post('/auth/validate', (req, res) => {
    const { token } = req.body;

    // Simulate auth validation with some latency
    const latency = Math.random() * 50 + 10;
    setTimeout(() => {
        console.log(JSON.stringify({
            timestamp: new Date().toISOString(),
            service: 'auth-service',
            level: 'INFO',
            message: `Token validation request`,
            duration_ms: latency
        }));

        res.json({ valid: true, user_id: Math.floor(Math.random() * 1000) });
    }, latency);
});

app.get('/health', (req, res) => {
    res.json({ status: 'healthy' });
});

const PORT = process.env.PORT || 8001;
app.listen(PORT, () => {
    console.log(`Auth service listening on port ${PORT}`);
});
