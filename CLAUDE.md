# memSTORY Project Development Guidelines

## 📋 Project Overview

### Project Name: memSTORY
- **Platform**: Android-based mobile application
- **Language**: English-based app (all UI text implemented in English)

### Core Objectives
1. **Personalized Second Brain**: On-device LLM trained with user's personal data
2. **VoiceChat Maximization**: Natural voice conversations to maximize user engagement time
3. **Complete Offline**: No internet required after initial runtime download
4. **Privacy Protection**: Personal data never leaves the device

### Ultimate Vision (Future Implementation)
- On-device LLM generates optimized prompts for effective communication with Universal LLMs
- User only needs to perform "simple agreement" to prompts written by local LLM
- Natural language feedback allows local LLM to improve prompts for Universal LLM communication

---

## 🛠️ Development Rules

### Technology Stack Restrictions
- **Dependency Injection**: Use Koin + KSP (Hilt prohibited)
- **UI Language**: Implement in English (Korean prohibited)
- **Documentation**: Generate Korean MD files upon Phase completion

### File Structure Rules
```
Project_A/
├── .files/
│   ├── bat_files/     # Batch files storage
│   ├── MD_files/      # Markdown documents storage  
│   └── APK_files/     # APK files storage
├── src/               # Android source code
└── CLAUDE.md          # This file
```

---

## 🔧 Git and GitHub Configuration

### Remote Repository Information
- **GitHub Repository**: https://github.com/peterjangminho/memSTORY_3
- **Personal Access Token**: `[USE_YOUR_TOKEN_HERE]`

### Git Push Strategy (Large File Handling)

#### 1. Pre-configuration (Execute before every push)
```bash
# Increase HTTP buffer size
git config http.postBuffer 524288000  # 500MB
git config http.maxRequestBuffer 100M
git config core.compression 0
```

#### 2. Staged Push Strategy
```bash
# Stage 1: Push text files first
git add *.md *.txt *.json *.xml *.gradle
git commit -m "docs: Add text files and configurations"
git push origin main

# Stage 2: Push source code  
git add src/ app/
git commit -m "feat: Add source code files"
git push origin main

# Stage 3: Push resource files
git add res/ assets/
git commit -m "res: Add resource and asset files" 
git push origin main

# Stage 4: Push large files (models, APKs, etc.)
git add .files/APK_files/ .files/models/
git commit -m "build: Add APK and model files"
git push origin main
```

#### 3. Error Response Strategy
```bash
# On push failure: Split into smaller units
git reset --soft HEAD~1  # Cancel last commit
git add [specific_files]  # Select specific files only
git commit -m "fix: Push specific files only"
git push origin main

# Individual file push
git add single_file.ext
git commit -m "fix: Add single file"
git push origin main
```

#### 4. Large File Management
```bash
# Temporarily exclude large files in .gitignore
echo "*.apk" >> .gitignore
echo "*.onnx" >> .gitignore  
echo "*.bin" >> .gitignore

# Re-add after push
git rm --cached .gitignore
# Modify .gitignore then
git add .gitignore
git commit -m "update: Modify gitignore for large files"
```

---

## 📦 Commit Message Rules

### Commit Types
- `feat:` New feature addition
- `fix:` Bug fix  
- `docs:` Documentation related
- `build:` Build related (APK, configurations)
- `res:` Resource files (images, audio, etc.)
- `refactor:` Code refactoring
- `test:` Test addition/modification
- `chore:` Other tasks

### Examples
```bash
git commit -m "feat: Implement TextChat UI with WhatsApp style"
git commit -m "fix: Resolve memory leak in LLM inference engine"  
git commit -m "build: Add Phase 0 APK with basic LLM integration"
git commit -m "docs: Update Phase 1 completion report in Korean"
```

---

## 🚀 Phase-based GitHub Management

### Required Tasks Upon Phase Completion
1. **Document Generation**: Save Korean MD files to `.files/MD_files/`
2. **APK Build**: Save Phase APK to `.files/APK_files/`  
3. **Commit & Push**: Apply staged push strategy
4. **Tag Creation**: Version tagging per Phase
```bash
git tag -a "phase-0" -m "Phase 0: Foundation & LLM Verification completed"
git push origin --tags
```

### Branch Strategy
```bash
# Main development
git checkout main

# Create Phase branch (if needed)
git checkout -b phase-1-textchat
# After work completion
git checkout main  
git merge phase-1-textchat
git branch -d phase-1-textchat
```

---

## ⚠️ Important Notes

### Security
- Do not include Personal Access Token in commits
- Exclude test data containing personal information

### Performance
- Consider Git LFS for large model files
- Monitor APK file size (recommended under 100MB)

### Backup
- Create local backup at important Phase completion points
- Separate storage for `.files/` directory

---

Follow these guidelines to ensure stable and systematic development.