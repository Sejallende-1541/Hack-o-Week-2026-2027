const express = require("express");

const app = express();

// Middleware to read JSON data
app.use(express.json());

// Import routes
const studentRoutes = require("./routes/studentRoutes");

// Use routes
app.use("/students", studentRoutes);

// Home route
app.get("/", (req, res) => {
    res.send("Student API is Running!");
});

// Start server
const PORT = 3000;

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});