# Frontend Setup - Quick Start

## 1. Install Node.js (if not already done)

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

## 2. Extract and Setup Frontend

```bash
cd /home/ubuntu/workspace/elbiat

# Extract the frontend
tar -xzvf frontend.tar.gz

# Install dependencies
cd frontend
npm install
```

## 3. Start Development Server

```bash
# Make sure your FastAPI is running on port 8000
npm run dev
```

Frontend will be at: **http://localhost:3000**

## 4. What You Get

| Page | URL | Description |
|------|-----|-------------|
| Dashboard | `/` | Overview + quick actions |
| Login | `/login` | JWT authentication |
| Register | `/register` | New user signup |
| Tasks | `/tasks` | List all benchmarks |
| Task Detail | `/tasks/charxiv` | Leaderboard + run evals |
| Chat | `/chat` | VLM chat with images |
| Gallery | `/gallery` | Image upload + management |

## 5. FastAPI Endpoints Needed

Make sure your FastAPI has these endpoints:

```
POST /api/auth/login        → { access_token, user }
POST /api/auth/register     → { access_token, user }
GET  /api/auth/me           → user object

GET  /api/evals/tasks       → list of tasks
GET  /api/evals/tasks/{name} → task detail
GET  /api/evals/tasks/{name}/leaderboard → leaderboard
POST /api/evals/tasks/{name}/runs → trigger eval
GET  /api/evals/models      → list of models

POST /api/chat              → streaming response
GET  /api/images            → list of images
POST /api/images/upload     → upload image
PATCH /api/images/{id}      → update visibility
```

## 6. Environment Variables (Optional)

Create `.env.local` in frontend folder:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 7. Build for Production

```bash
npm run build
npm start
```

---

## File Structure

```
frontend/
├── app/
│   ├── layout.tsx          # Root layout
│   ├── providers.tsx       # Auth + React Query
│   ├── page.tsx            # Dashboard
│   ├── login/page.tsx
│   ├── register/page.tsx
│   ├── chat/page.tsx
│   ├── gallery/page.tsx
│   └── tasks/
│       ├── page.tsx        # Task list
│       └── [name]/page.tsx # Task detail + leaderboard
├── components/
│   ├── ui/                 # Reusable components
│   └── layout/navbar.tsx
├── lib/
│   ├── api.ts              # API client
│   ├── auth.ts             # Auth store (Zustand)
│   └── utils.ts
└── package.json
```

## Customization

- **Colors**: Edit `tailwind.config.ts`
- **Components**: All in `components/ui/`
- **API Client**: Edit `lib/api.ts` to match your endpoints
