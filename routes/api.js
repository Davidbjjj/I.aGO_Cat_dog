const express = require('express');
const multer = require('multer');
const path = require('path');
const router = express.Router();

// Configuração do multer para upload de arquivos
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/');
    },
    filename: function (req, file, cb) {
        cb(null, Date.now() + '-' + file.originalname);
    }
});

const upload = multer({
    storage: storage,
    fileFilter: function (req, file, cb) {
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Apenas imagens são permitidas!'), false);
        }
    },
    limits: {
        fileSize: 5 * 1024 * 1024 // 5MB
    }
});

// Endpoint para health check da API
router.get('/health', (req, res) => {
    res.json({ 
        status: 'API funcionando', 
        timestamp: new Date().toISOString() 
    });
});

// Endpoint para informações do modelo
router.get('/model-info', (req, res) => {
    res.json({
        model: 'Classificador Gatos vs Cachorros',
        version: '1.0.0',
        classes: ['cat', 'dog'],
        description: 'Modelo de classificação de imagens usando TensorFlow.js'
    });
});

// Endpoint para upload e classificação de imagem
router.post('/classify', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'Nenhuma imagem enviada' });
        }

        // Simulação de classificação (substitua pela lógica real do TensorFlow)
        const classificationResult = {
            predictions: [
                {
                    className: 'cat',
                    probability: Math.random() * 0.5 + 0.3,
                    boundingBox: null
                },
                {
                    className: 'dog',
                    probability: Math.random() * 0.5 + 0.3,
                    boundingBox: null
                }
            ],
            success: true,
            timestamp: new Date().toISOString()
        };

        // Encontrar a classe com maior probabilidade
        const highestPrediction = classificationResult.predictions.reduce((prev, current) => {
            return (prev.probability > current.probability) ? prev : current;
        });

        res.json({
            success: true,
            classification: highestPrediction.className,
            confidence: (highestPrediction.probability * 100).toFixed(2),
            allPredictions: classificationResult.predictions.map(p => ({
                class: p.className,
                confidence: (p.probability * 100).toFixed(2)
            })),
            imageUrl: `/uploads/${req.file.filename}`
        });

    } catch (error) {
        console.error('Erro na classificação:', error);
        res.status(500).json({ 
            error: 'Erro interno no servidor',
            message: error.message 
        });
    }
});

// Endpoint para listar uploads recentes
router.get('/recent-classifications', (req, res) => {
    // Simulação de dados recentes
    const recentData = [
        {
            id: 1,
            className: 'cat',
            confidence: '87.45%',
            timestamp: new Date().toISOString(),
            image: '/uploads/sample-cat.jpg'
        },
        {
            id: 2,
            className: 'dog',
            confidence: '92.13%',
            timestamp: new Date(Date.now() - 3600000).toISOString(),
            image: '/uploads/sample-dog.jpg'
        }
    ];
    
    res.json(recentData);
});

module.exports = router;