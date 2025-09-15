
document.addEventListener('DOMContentLoaded', () => {
    const API_URL = '/api';

    // Screens
    const screens = document.querySelectorAll('.screen');
    const navButtons = document.querySelectorAll('.nav-btn');

    // State
    let currentLessonId = null;
    let currentLessonData = null; // Holds the data for the currently open lesson

    // Elements
    const newLessonBtn = document.getElementById('new-lesson-btn');
    const lessonsList = document.getElementById('lessons-list');
    const lessonTitleInput = document.getElementById('lesson-title-input');
    const lessonContentView = document.getElementById('lesson-content-view');
    const lessonContentTextarea = document.getElementById('lesson-content-textarea');
    const editLessonBtn = document.getElementById('edit-lesson-btn');
    const saveLessonBtn = document.getElementById('save-lesson-btn');
    const generatePromptInput = document.getElementById('generate-prompt-input');
    const generateBtn = document.getElementById('generate-btn');
    const sourcesList = document.getElementById('sources-list');
    const addSourceBtn = document.getElementById('add-source-btn');
    const uploadModal = document.getElementById('upload-modal');
    const cancelUploadBtn = document.getElementById('cancel-upload-btn');
    const startUploadBtn = document.getElementById('start-upload-btn');
    const fileInput = document.getElementById('file-input');
    const uploadProgressContainer = document.getElementById('upload-progress-container');
    const loadingOverlay = document.getElementById('loading-overlay');
    const generateFullLessonBtn = document.getElementById('generate-full-lesson-btn');

    // --- Navigation ---
    navButtons.forEach(button => {
        button.addEventListener('click', () => {
            const screenId = button.dataset.screen;
            showScreen(screenId);
        });
    });

    function showScreen(screenId) {
        screens.forEach(screen => {
            screen.classList.toggle('active', screen.id === screenId);
            screen.classList.toggle('hidden', screen.id !== screenId);
        });
        navButtons.forEach(btn => {
            btn.classList.toggle('text-purple-600', btn.dataset.screen === screenId);
        });
    }

    // --- Loading Overlay ---
    function showLoading(message = 'Generating lesson...') {
        loadingOverlay.querySelector('p').textContent = message;
        loadingOverlay.classList.remove('hidden');
    }

    function hideLoading() {
        loadingOverlay.classList.add('hidden');
    }

    // --- Markdown Rendering ---
    function renderMarkdown(md) {
        if (md) {
            lessonContentView.innerHTML = marked.parse(md);
        } else {
            lessonContentView.innerHTML = '';
        }
    }

    // --- API Functions ---
    async function getLessons() {
        showLoading('Loading lessons...');
        try {
            const response = await fetch(`${API_URL}/lessons`);
            const lessons = await response.json();
            renderLessons(lessons);
        } catch (error) {
            console.error('Error fetching lessons:', error);
        } finally {
            hideLoading();
        }
    }

    async function createLesson() {
        const title = prompt('Enter lesson title:');
        if (!title) return;

        showLoading('Creating lesson...');
        try {
            const response = await fetch(`${API_URL}/lessons`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title })
            });
            const newLesson = await response.json();
            currentLessonId = newLesson.id;
            await getLessons();
            openLesson(currentLessonId);
        } catch (error) {
            console.error('Error creating lesson:', error);
        } finally {
            hideLoading();
        }
    }

    async function openLesson(lessonId) {
        currentLessonId = lessonId;
        showLoading('Opening lesson...');
        try {
            const response = await fetch(`${API_URL}/lessons/${lessonId}`);
            currentLessonData = await response.json();
            lessonTitleInput.value = currentLessonData.title;
            lessonContentTextarea.value = currentLessonData.content;
            renderMarkdown(currentLessonData.content);
            renderSources(currentLessonData.sources);
            showScreen('workspace-screen');
            switchToViewMode();
        } catch (error) {
            console.error('Error opening lesson:', error);
        } finally {
            hideLoading();
        }
    }

    async function saveLesson() {
        if (!currentLessonId) return;
        showLoading('Saving lesson...');
        try {
            const newContent = lessonContentTextarea.value;
            await fetch(`${API_URL}/lessons/${currentLessonId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    title: lessonTitleInput.value,
                    content: newContent
                })
            });
            currentLessonData.content = newContent;
            renderMarkdown(newContent);
            switchToViewMode();
        } catch (error) {
            console.error('Error saving lesson:', error);
        } finally {
            hideLoading();
        }
    }

    async function generateContent() {
        if (!currentLessonId) return;
        const prompt = generatePromptInput.value;
        if (!prompt) return;

        showLoading('Generating content...');
        try {
            const response = await fetch(`${API_URL}/lessons/${currentLessonId}/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt })
            });
            const data = await response.json();
            currentLessonData.content = data.content;
            lessonContentTextarea.value = data.content;
            renderMarkdown(data.content);
            generatePromptInput.value = '';
            switchToViewMode();
        } catch (error) {
            console.error('Error generating content:', error);
        } finally {
            hideLoading();
        }
    }

    async function getSources() {
        if (!currentLessonId) return;
        try {
            const response = await fetch(`${API_URL}/lessons/${currentLessonId}/sources`);
            const sources = await response.json();
            renderSources(sources);
        } catch (error) {
            console.error('Error fetching sources:', error);
        }
    }

    function uploadFiles() {
        if (!currentLessonId) return;
        const files = fileInput.files;
        if (files.length === 0) return;

        Array.from(files).forEach(file => {
            const formData = new FormData();
            formData.append('file', file);

            const xhr = new XMLHttpRequest();
            xhr.open('POST', `${API_URL}/lessons/${currentLessonId}/sources`, true);

            const progressId = `progress-${Date.now()}`;
            const progressElement = document.createElement('div');
            progressElement.innerHTML = `
                <p>${file.name}</p>
                <div class="w-full bg-gray-200 rounded-full h-2.5">
                    <div id="${progressId}" class="bg-purple-600 h-2.5 rounded-full" style="width: 0%"></div>
                </div>
            `;
            uploadProgressContainer.appendChild(progressElement);

            xhr.upload.onprogress = (event) => {
                if (event.lengthComputable) {
                    const percentComplete = (event.loaded / event.total) * 100;
                    document.getElementById(progressId).style.width = `${percentComplete}%`;
                }
            };

            xhr.onload = () => {
                if (xhr.status === 201) {
                    getSources();
                } else {
                    console.error('Upload failed:', xhr.responseText);
                }
            };

            xhr.send(formData);
        });

        closeUploadModal();
    }
    
    async function ingestSources() {
        if (!currentLessonId) return;
        showLoading('Ingesting sources...');
        try {
            await fetch(`${API_URL}/lessons/${currentLessonId}/ingest`, { method: 'POST' });
            await getSources();
        } catch (error) {
            console.error('Error ingesting sources:', error);
        } finally {
            hideLoading();
        }
    }

    async function deleteSource(sourceId) {
        if (!currentLessonId || !sourceId) return;
        if (!confirm('Are you sure you want to delete this source?')) return;

        showLoading('Deleting source...');
        try {
            const response = await fetch(`${API_URL}/lessons/${currentLessonId}/sources/${sourceId}`, {
                method: 'DELETE'
            });
            if (!response.ok) {
                throw new Error('Failed to delete source');
            }
            await getSources(); // Refresh the list
        } catch (error) {
            console.error('Error deleting source:', error);
        } finally {
            hideLoading();
        }
    }

    async function generateFullLesson() {
        if (!currentLessonId) return;
        if (!confirm('This will generate a new lesson note from all ingested sources, potentially overwriting existing content. Proceed?')) return;

        const lessonTitle = lessonTitleInput.value || 'the lesson';
        const comprehensivePrompt = `
            You are an expert curriculum designer and educator.
            Based on the comprehensive context provided below, which contains all the raw materials for a lesson, your task is to synthesize this information into a complete, detailed, and engaging lesson note.
            The lesson is titled "${lessonTitle}".

            Your output should be well-structured and formatted using Markdown. It must include the following sections:
            1.  **Lesson Title**: The title of the lesson.
            2.  **Overview & Learning Objectives**: A brief summary of the lesson and a bulleted list of what students will be able to do after completing it.
            3.  **Key Concepts**: A detailed explanation of the core concepts, broken down into logical sub-sections. Use headings, lists, and bold text to ensure clarity.
            4.  **Activities & Exercises**: Suggest at least two practical activities or exercises that reinforce the key concepts. Provide clear instructions for each.
            5.  **Assessment**: Propose a method for assessing student understanding (e.g., a short quiz with questions and answers, a project idea).
            6.  **Conclusion**: A concluding summary of the lesson's main points.

            Synthesize the provided context thoroughly. Do not simply copy sections. Create a cohesive and pedagogically sound lesson note.

            Important: Your final output must be only the raw Markdown content for the lesson note. Do not wrap it in a code block or use the ` + '```markdown fences.'
        ;

        showLoading('Generating full lesson...');
        try {
            const response = await fetch(`${API_URL}/lessons/${currentLessonId}/generate-full`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: comprehensivePrompt })
            });
            const data = await response.json();
            currentLessonData.content = data.content;
            lessonContentTextarea.value = data.content;
            renderMarkdown(data.content);
            showScreen('workspace-screen');
            switchToViewMode();
        } catch (error) {
            console.error('Error generating full lesson:', error);
        } finally {
            hideLoading();
        }
    }

    // --- Rendering ---
    function renderLessons(lessons) {
        lessonsList.innerHTML = '';
        lessons.forEach(lesson => {
            const lessonCard = document.createElement('div');
            lessonCard.className = 'bg-white p-4 rounded-2xl shadow-md cursor-pointer';
            lessonCard.innerHTML = `<h3 class="text-xl font-bold">${lesson.title}</h3>`;
            lessonCard.addEventListener('click', () => openLesson(lesson.id));
            lessonsList.appendChild(lessonCard);
        });
    }

    function renderSources(sources) {
        sourcesList.innerHTML = '';
        sources.forEach(source => {
            const sourceCard = document.createElement('div');
            sourceCard.className = 'bg-white p-4 rounded-2xl shadow-md';
            sourceCard.innerHTML = `
                <div class="flex justify-between items-center">
                    <p class="font-bold truncate">${source.filename}</p>
                    <button data-id="${source.id}" class="delete-source-btn text-red-500 hover:text-red-700 ml-2">
                        <span class="material-icons">delete</span>
                    </button>
                </div>
                <p class="text-sm text-gray-500">${source.status}</p>
            `;
            sourcesList.appendChild(sourceCard);
        });

        // Add event listeners to delete buttons
        document.querySelectorAll('.delete-source-btn').forEach(button => {
            button.addEventListener('click', (e) => {
                e.stopPropagation();
                const sourceId = e.currentTarget.dataset.id;
                deleteSource(sourceId);
            });
        });
        
        if (sources.some(s => s.status === 'uploaded')) {
            const ingestBtn = document.createElement('button');
            ingestBtn.className = 'bg-blue-500 text-white px-4 py-2 rounded-2xl mt-4';
            ingestBtn.textContent = 'Ingest Sources';
            ingestBtn.onclick = ingestSources;
            sourcesList.appendChild(ingestBtn);
        }
    }

    // --- View/Edit Mode ---
    function switchToViewMode() {
        lessonContentView.classList.remove('hidden');
        lessonContentTextarea.classList.add('hidden');
        editLessonBtn.classList.remove('hidden');
        saveLessonBtn.classList.add('hidden');
    }

    function switchToEditMode() {
        if (!currentLessonData) return;
        lessonContentTextarea.value = currentLessonData.content;
        lessonContentView.classList.add('hidden');
        lessonContentTextarea.classList.remove('hidden');
        editLessonBtn.classList.add('hidden');
        saveLessonBtn.classList.remove('hidden');
    }

    // --- Modals ---
    function showUploadModal() {
        uploadModal.classList.remove('hidden');
    }

    function closeUploadModal() {
        uploadModal.classList.add('hidden');
        fileInput.value = '';
        uploadProgressContainer.innerHTML = '';
    }

    // --- Event Listeners ---
    newLessonBtn.addEventListener('click', createLesson);
    lessonTitleInput.addEventListener('blur', saveLesson);
    editLessonBtn.addEventListener('click', switchToEditMode);
    saveLessonBtn.addEventListener('click', saveLesson);
    generateBtn.addEventListener('click', generateContent);
    addSourceBtn.addEventListener('click', showUploadModal);
    cancelUploadBtn.addEventListener('click', closeUploadModal);
    startUploadBtn.addEventListener('click', uploadFiles);
    generateFullLessonBtn.addEventListener('click', generateFullLesson);

    // --- Initial Load ---
    getLessons();
    showScreen('highlights-screen');
});
