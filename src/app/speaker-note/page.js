// app/speaker-notes/page.js
import fs from 'fs';
import path from 'path';
import ReactMarkdown from 'react-markdown';

async function getMarkdownContent() {
    const filePath = path.join(process.cwd(), 'public', 'llm_workshop_speaker_notes.md');
    const content = fs.readFileSync(filePath, 'utf8');
    return content;
}

export default async function SpeakerNotesPage() {
    const markdown = await getMarkdownContent();

    return (
        <div className="max-w-4xl mx-auto py-8 px-4">
            <h1 className="text-3xl font-bold mb-6">Speaker Notes</h1>
            <ReactMarkdown>{markdown}</ReactMarkdown>
        </div>
    );
}
