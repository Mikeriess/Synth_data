let conversations = [];
let currentIndex = 0;
let evaluations = {};

// Load conversations from JSON file
async function loadConversations() {
    try {
        const response = await fetch('conversations.json');
        conversations = await response.json();
        loadEvaluations();
        
        // Check for conversation ID in URL
        const urlParams = new URLSearchParams(window.location.search);
        const conversationId = urlParams.get('id');
        if (conversationId) {
            loadConversationById(conversationId);
        } else {
            updateDisplay();
        }
    } catch (error) {
        console.error('Error loading conversations:', error);
    }
}

// Load evaluations from localStorage
function loadEvaluations() {
    const savedEvals = localStorage.getItem('conversation_evaluations');
    if (savedEvals) {
        evaluations = JSON.parse(savedEvals);
    }
}

// Save evaluation to localStorage
function saveEvaluation() {
    const currentNote = document.getElementById('conversation-notes').value;
    const languageRating = document.querySelector('input[name="language"]:checked')?.value || '';
    const factualityRating = document.querySelector('input[name="factuality"]:checked')?.value || '';
    const knowledgeRating = document.querySelector('input[name="knowledge"]:checked')?.value || '';
    const similarityRating = document.querySelector('input[name="similarity"]:checked')?.value || '';
    const notQA = document.getElementById('not-qa').checked;
    const invalidGen = document.getElementById('invalid-gen').checked;
    const incompleteOrig = document.getElementById('incomplete-orig').checked;
    
    const conversationId = conversations[currentIndex].conversation_id;
    
    const evaluation = {
        conversation_id: conversationId,
        timestamp: new Date().toISOString(),
        notes: currentNote,
        language: languageRating,
        factuality: factualityRating,
        knowledge: knowledgeRating,
        similarity: similarityRating,
        not_qa: notQA,
        invalid_gen: invalidGen,
        incomplete_orig: incompleteOrig
    };
    
    // Save to localStorage for UI state
    evaluations[conversationId] = evaluation;
    localStorage.setItem('conversation_evaluations', JSON.stringify(evaluations));
    
    // Save to annotations.json file
    saveToAnnotationsFile(evaluation);
    
    // Visual feedback
    const saveButton = document.querySelector('.save-note button');
    const originalText = saveButton.textContent;
    saveButton.textContent = 'Saved!';
    setTimeout(() => {
        saveButton.textContent = originalText;
    }, 1000);
}

async function saveToAnnotationsFile(evaluation) {
    try {
        const response = await fetch('/save_annotation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(evaluation)
        });
        
        if (!response.ok) {
            throw new Error('Failed to save annotation');
        }
    } catch (error) {
        console.error('Error saving annotation:', error);
        alert('Failed to save annotation to file. Your evaluation is still saved in the browser.');
    }
}

function updateDisplay() {
    const conversation = conversations[currentIndex];
    if (!conversation) return;

    // Load any existing evaluation for this conversation
    const conversationId = conversation.conversation_id;
    const evaluation = evaluations[conversationId] || {};
    
    // Update notes and ratings
    document.getElementById('conversation-notes').value = evaluation.notes || '';
    
    // Update checkboxes
    document.getElementById('not-qa').checked = evaluation.not_qa || false;
    document.getElementById('invalid-gen').checked = evaluation.invalid_gen || false;
    document.getElementById('incomplete-orig').checked = evaluation.incomplete_orig || false;
    
    // Clear all radio selections first
    document.querySelectorAll('input[type="radio"]').forEach(radio => radio.checked = false);
    
    // Set saved ratings
    if (evaluation.language) {
        document.querySelector(`input[name="language"][value="${evaluation.language}"]`).checked = true;
    }
    if (evaluation.factuality) {
        document.querySelector(`input[name="factuality"][value="${evaluation.factuality}"]`).checked = true;
    }
    if (evaluation.knowledge) {
        document.querySelector(`input[name="knowledge"][value="${evaluation.knowledge}"]`).checked = true;
    }

    // Update conversation ID
    document.getElementById('conversation-id').textContent = 
        `Conversation ID: ${conversation.conversation_id}`;

    // Update conversation count
    document.getElementById('conversation-count').textContent = 
        `Conversation ${currentIndex + 1} of ${conversations.length}`;

    // Display original messages
    const originalDiv = document.getElementById('original-messages');
    if (Array.isArray(conversation.orig_messages)) {
        originalDiv.innerHTML = conversation.orig_messages
            .map(msg => createMessageHTML(msg))
            .join('');
    } else {
        console.error('Original messages not in expected format:', conversation.orig_messages);
    }

    // Display synthetic messages
    const syntheticDiv = document.getElementById('synthetic-messages');
    if (Array.isArray(conversation.synthetic_messages)) {
        syntheticDiv.innerHTML = conversation.synthetic_messages
            .map(msg => createMessageHTML(msg, true))
            .join('');
    } else {
        console.error('Synthetic messages not in expected format:', conversation.synthetic_messages);
    }
}

function createMessageHTML(message, isSynthetic = false) {
    if (!message || typeof message !== 'object') {
        console.error('Invalid message format:', message);
        return '<div class="message"><p>Error: Invalid message format</p></div>';
    }

    // Get the user ID from either 'user' or 'poster_id' field
    let userId = message.user || message.poster_id || 'Unknown';
    
    // Convert float user IDs to integers if needed
    if (typeof userId === 'number') {
        userId = Math.floor(userId);
    }

    return `
        <div class="message ${isSynthetic ? 'synthetic-message' : ''}">
            <strong>Person ${userId}</strong>
            <p>${message.text || message.content || 'No content available'}</p>
        </div>
    `;
}

function nextConversation() {
    if (currentIndex < conversations.length - 1) {
        currentIndex++;
        updateDisplay();
    }
}

function previousConversation() {
    if (currentIndex > 0) {
        currentIndex--;
        updateDisplay();
    }
}

function randomConversation() {
    if (conversations.length > 0) {
        // Generate random index different from current
        let newIndex;
        do {
            newIndex = Math.floor(Math.random() * conversations.length);
        } while (conversations.length > 1 && newIndex === currentIndex);
        
        currentIndex = newIndex;
        updateDisplay();
    }
}

function jumpToIndex() {
    const input = document.getElementById('jump-index');
    const targetIndex = parseInt(input.value) - 1; // Convert to 0-based index
    
    if (isNaN(targetIndex)) {
        input.value = '';
        return;
    }
    
    // Bound check
    if (targetIndex >= 0 && targetIndex < conversations.length) {
        currentIndex = targetIndex;
        updateDisplay();
    } else {
        // Reset input if out of bounds
        input.value = '';
    }
}

// Load specific conversation by ID
function loadConversationById(id) {
    const index = conversations.findIndex(conv => conv.conversation_id === id);
    if (index !== -1) {
        currentIndex = index;
        updateDisplay();
    }
}

// Initialize
loadConversations();

// Function to load conversation data
function loadConversation(conversationId) {
    fetch(`/conversation?id=${conversationId}`)
        .then(response => response.json())
        .then(data => {
            // Clear previous conversation
            conversationContainer.innerHTML = '';
            
            // Store current conversation ID
            currentConversationId = conversationId;
            
            // Display original messages
            const originalMessages = document.createElement('div');
            originalMessages.className = 'message-column';
            originalMessages.innerHTML = `<h2>Original Messages</h2>`;
            
            data.orig_messages.forEach(msg => {
                const msgElement = createMessageElement(msg, 'original');
                originalMessages.appendChild(msgElement);
            });
            
            // Display synthetic messages
            const syntheticMessages = document.createElement('div');
            syntheticMessages.className = 'message-column';
            syntheticMessages.innerHTML = `<h2>Synthetic Messages</h2>`;
            
            // Check if we have synthetic_messages or synth_messages
            const synthMsgs = data.synthetic_messages || data.synth_messages || [];
            
            synthMsgs.forEach(msg => {
                const msgElement = createMessageElement(msg, 'synthetic');
                syntheticMessages.appendChild(msgElement);
            });
            
            // Add both columns to the container
            conversationContainer.appendChild(originalMessages);
            conversationContainer.appendChild(syntheticMessages);
            
            // Update annotation if it exists
            updateAnnotationDisplay(conversationId);
        })
        .catch(error => {
            console.error('Error loading conversation:', error);
            conversationContainer.innerHTML = `<p class="error">Error loading conversation: ${error.message}</p>`;
        });
} 