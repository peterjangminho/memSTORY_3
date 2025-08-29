# memSTORY Development Plan (English)

## 📋 Project Overview

### Project Name: memSTORY
**Mission**: Android-based Second Brain app powered by personalized on-device LLM

### Core Objectives
1. **Personalized AI**: On-device LLM trained with user's personal data
2. **VoiceChat Maximization**: Natural voice conversations to maximize user engagement time
3. **Complete Offline**: No internet required after initial download
4. **Privacy Protection**: Personal data never leaves the device

### Ultimate Vision
On-device LLM generates optimized prompts for effective communication with Universal LLMs

---

## ⚠️ Analysis of Previous Failures & Mitigation Strategies

### Major Failure Factors
1. **Simultaneous complex feature development** → **Phased development approach**
2. **Technical challenges of on-device LLM integration** → **Risk-first validation**
3. **Over-complex architecture** → **Start minimal, expand gradually**
4. **Android-specific technical gaps** → **Build solid foundations first**

### Key Success Strategies
- **Walking Skeleton**: Implement simplest end-to-end functionality first
- **Risk-First**: Validate most difficult parts (LLM) early
- **Independent Modules**: Each component testable in isolation
- **APK Validation**: Verify actual device functionality at each stage

---

## 🏗️ System Architecture

### Technology Stack

#### Core Technologies
- **Platform**: Android (Kotlin)
- **UI**: Jetpack Compose
- **Dependency Injection**: Koin + KSP
- **Architecture**: Clean Architecture (Presentation-Domain-Data)

#### AI & ML Stack
- **On-device LLM**: Compatible LLM with quantized optimization
- **STT**: Vosk (vosk-model-small-en-us-zamia-0.5)
- **TTS**: Piper (en_US-lessac-medium)
- **Voice Processing**: WebRTC (VAD + AEC + NS)
- **Emotion Recognition**: VADER

#### Memory & RAG System
- **Embedding Model**: Lightweight embedding solution
- **Vector Storage**: SQLite with vector capabilities
- **Memory Management**: Custom RAG system

#### Security
- **Encryption**: Android KeyStore + AES-256 + Argon2
- **Data Protection**: App-internal encrypted storage

### System Architecture Diagram

```
┌─────────────────────────────────────────────┐
│                Presentation Layer            │
│  ┌─────────────┬─────────────┬─────────────┐ │
│  │ VoiceChat   │ TextChat    │ Settings    │ │
│  │ UI          │ UI          │ UI          │ │
│  └─────────────┴─────────────┴─────────────┘ │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│                Domain Layer                  │
│  ┌─────────────┬─────────────┬─────────────┐ │
│  │ VoiceChat   │ Memory      │ LLM         │ │
│  │ UseCase     │ UseCase     │ UseCase     │ │
│  └─────────────┴─────────────┴─────────────┘ │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│                Data Layer                    │
│  ┌─────────────┬─────────────┬─────────────┐ │
│  │ LLM         │ Voice       │ Memory      │ │
│  │ Repository  │ Repository  │ Repository  │ │
│  └─────────────┴─────────────┴─────────────┘ │
└─────────────────────────────────────────────┘
┌─────────────────────────────────────────────┐
│              Infrastructure Layer            │
│  ┌─────────────┬─────────────┬─────────────┐ │
│  │ LLM         │ WebRTC      │ SQLite +    │ │
│  │ Runtime     │ Pipeline    │ Vector DB   │ │
│  └─────────────┴─────────────┴─────────────┘ │
└─────────────────────────────────────────────┘
```

---

## 📊 Risk Analysis & Mitigation

### 🔴 HIGH RISK (Priority Validation Required)
1. **On-device LLM Integration**
   - Risk: Memory shortage, performance degradation, battery drain
   - Mitigation: Start with smaller models, apply quantization, optimize memory

2. **Real-time Voice Processing**
   - Risk: Latency, audio quality degradation, interruptions
   - Mitigation: WebRTC validation, buffer optimization, phased implementation

### 🟡 MEDIUM RISK
3. **RAG Memory System**
   - Risk: Search accuracy, storage space, performance
   - Mitigation: Start with simple structure, progressive improvement

4. **User Permission Management**
   - Risk: Permission denial, app functionality limitation
   - Mitigation: Prepare alternative scenarios per permission

### 🟢 LOW RISK
5. **UI/UX Implementation**
   - Risk: Usability issues
   - Mitigation: Start with simple design

---

## 🚀 Phase-by-Phase Development Roadmap

### Phase 0: Foundation & LLM Verification (Week 1-2)
**Goal**: Basic project structure + minimal on-device LLM validation

#### Milestones:
- [ ] M0.1: Android project setup
- [ ] M0.2: Basic Compose UI structure
- [ ] M0.3: Koin DI configuration
- [ ] M0.4: LLM runtime integration
- [ ] M0.5: Minimal LLM model loading test
- [ ] M0.6: "Hello World" text generation verification
- [ ] M0.7: APK build and actual device testing

#### Detailed To-Do:
**T0.1: Project Setup**
- [ ] Create Android Studio project (API 26+)
- [ ] Gradle configuration (Kotlin, Compose, Koin)
- [ ] Create basic package structure
- [ ] Git initialization and .gitignore setup

**T0.2: UI Foundation**
- [ ] MainActivity + Compose setup
- [ ] Navigation Component configuration
- [ ] Basic theme and color definition (pastel tones)
- [ ] Simple "Hello World" screen

**T0.3: Dependency Injection**
- [ ] Koin module structure design
- [ ] Basic AppModule, DataModule setup
- [ ] Application class configuration

**T0.4: LLM Runtime Integration**
- [ ] Add appropriate LLM runtime library
- [ ] Download and prepare test model
- [ ] Place model in assets folder
- [ ] Write basic model loading code

**T0.5: LLM Inference Engine**
- [ ] Create LLM inference engine class
- [ ] Integrate tokenizer
- [ ] Implement basic text generation function
- [ ] Basic memory management implementation

**T0.6: Validation & Testing**
- [ ] Write unit tests
- [ ] "Hello World" generation test
- [ ] Create APK build script
- [ ] Verify functionality on actual device

**Success Criteria**: 
- APK builds successfully
- LLM generates "Hello World" text on actual device
- Memory usage within acceptable range

---

### Phase 1: Core LLM Integration & TextChat (Week 3-4)
**Goal**: Complete LLM integration and basic TextChat implementation

#### Milestones:
- [ ] M1.1: LLM loading optimization (lazy loading, memory management)
- [ ] M1.2: Context window management implementation
- [ ] M1.3: TextChat UI implementation (WhatsApp style)
- [ ] M1.4: Conversation history save/load
- [ ] M1.5: Performance optimization
- [ ] M1.6: APK build and performance validation

#### Detailed To-Do:
**T1.1: LLM Engine Optimization**
- [ ] Implement lazy loading
- [ ] Apply memory optimization
- [ ] Model unload/reload logic
- [ ] Context window management
- [ ] Token counting system

**T1.2: TextChat UI**
- [ ] Implement ChatScreen Compose
- [ ] Define Message data class
- [ ] Use LazyColumn for message display
- [ ] User/AI message distinction UI
- [ ] Input field and send button

**T1.3: Conversation Logic**
- [ ] Implement ChatRepository
- [ ] Conversation session management
- [ ] Message history SQLite storage
- [ ] Context retention logic

**T1.4: Performance Optimization**
- [ ] Hardware acceleration configuration (if available)
- [ ] CPU optimization
- [ ] Batch size optimization
- [ ] Background thread processing

**Success Criteria**:
- Complete TextChat with continuous conversation capability
- Response time under 3 seconds
- 30 minutes continuous use without app crashes

---

### Phase 2: Voice Processing Pipeline (Week 5-6)
**Goal**: STT, TTS, and voice processing pipeline implementation

#### Milestones:
- [ ] M2.1: Vosk STT integration
- [ ] M2.2: Piper TTS integration  
- [ ] M2.3: WebRTC voice processing (VAD, AEC, NS)
- [ ] M2.4: Audio recording/playback management
- [ ] M2.5: Permission management (microphone)
- [ ] M2.6: Basic voice input/output testing

#### Detailed To-Do:
**T2.1: STT System**
- [ ] Integrate Vosk Android library
- [ ] Add vosk-model-small-en-us-zamia-0.5 model to assets
- [ ] Implement VoskSTTEngine class
- [ ] Real-time speech recognition implementation
- [ ] Recognition result callback system

**T2.2: TTS System**
- [ ] Integrate Piper binary and models
- [ ] Implement PiperTTSEngine class
- [ ] Text → speech conversion implementation
- [ ] Audio playback management

**T2.3: WebRTC Voice Processing**
- [ ] Integrate WebRTC library
- [ ] Implement VAD (Voice Activity Detection)
- [ ] Implement AEC (Acoustic Echo Cancellation)
- [ ] Implement NS (Noise Suppression)
- [ ] Real-time audio processing pipeline

**T2.4: Audio System**
- [ ] AudioRecord management
- [ ] AudioTrack management
- [ ] Audio format optimization (16kHz, 16-bit)
- [ ] Buffer and latency optimization

**T2.5: Permission Management**
- [ ] Microphone permission request
- [ ] Permission denial alternative flow
- [ ] Permission status monitoring

**Success Criteria**:
- Voice → text → voice conversion completed within 3 seconds
- Clear audio quality (noise reduction effect confirmed)
- No memory leaks during continuous voice processing

---

### Phase 3: VoiceChat Engine Implementation (Week 7-8)
**Goal**: Conversation state machine and VoiceChat logic implementation

#### Milestones:
- [ ] M3.1: Conversation state machine design and implementation
- [ ] M3.2: Emotion recognition system integration
- [ ] M3.3: User interruption logic implementation
- [ ] M3.4: VoiceChat UI implementation (circle + wave effect)
- [ ] M3.5: Stage-based prompt system
- [ ] M3.6: Complete VoiceChat flow testing

#### Detailed To-Do:
**T3.1: Conversation State Machine**
- [ ] Define ConversationState enumeration
- [ ] Implement state transition logic
  - Opening → Emotion Expansion → Free Speech → Silence → Re-stimulation
- [ ] Timer-based state transitions
- [ ] User event-based state transitions

**T3.2: Emotion Recognition System**
- [ ] Integrate VADER sentiment analysis library
- [ ] Calculate emotion scores from voice text
- [ ] Emotion-based response generation logic
- [ ] Emotion history tracking

**T3.3: User Interruption**
- [ ] VAD monitoring during TTS playback
- [ ] User voice detection after TTS removal via AEC
- [ ] Immediate TTS stop upon interruption detection
- [ ] Automatic switch to STT mode

**T3.4: VoiceChat UI**
- [ ] Central circular button UI (Compose)
- [ ] Wave animation during voice recognition
- [ ] Bottom conversation text display (3-line limit)
- [ ] Visual conversation state indication

**T3.5: Prompt System**
- [ ] Opening system prompt (within 200 tokens)
- [ ] Emotion expansion prompt generation
- [ ] Narrowing Principle application logic
- [ ] Re-stimulation question generation (Question A/B)

**T3.6: Response System**
- [ ] Short responses like "do you?", "really?"
- [ ] Appropriate timing responses after 2-3 sentences
- [ ] Summary-type responses for 1+ minute conversations
- [ ] Responses without User Interruption application

**Success Criteria**:
- Natural progression of complete conversation cycle
- Immediate response to user interruption (within 1 second)
- Appropriate responses based on emotions
- 30 minutes of uninterrupted conversation capability

---

### Phase 4: Memory Generation Engine (Week 9-10)
**Goal**: RAG-based memory generation and management system

#### Milestones:
- [ ] M4.1: Embedding system implementation
- [ ] M4.2: Vector database construction
- [ ] M4.3: Idle time #1 - Calendar/memo data collection
- [ ] M4.4: Idle time #2 - All-inData generation
- [ ] M4.5: Keyinfo mark system
- [ ] M4.6: Memory search and context generation

#### Detailed To-Do:
**T4.1: Embedding System**
- [ ] Implement lightweight embedding model
- [ ] Implement EmbeddingEngine class
- [ ] Text → vector conversion functionality
- [ ] Similarity calculation function

**T4.2: Vector Database**
- [ ] Setup SQLite with vector capabilities
- [ ] Design vector storage schema
- [ ] Implement VectorRepository class
- [ ] Optimize similarity search queries

**T4.3: Data Collection System**
- [ ] Calendar app access permission
- [ ] Implement CalendarDataCollector
- [ ] Read memo app data (if available)
- [ ] Structure data by date

**T4.4: All-inData Generation**
- [ ] Data grouping algorithm
- [ ] Topic-based clustering
- [ ] Summary generation (using LLM)
- [ ] Time-based weighting application

**T4.5: Keyinfo Mark**
- [ ] Conversation time tracking
- [ ] Importance score calculation
- [ ] Keyinfo marking logic
- [ ] Priority sorting system

**T4.6: Background Processing**
- [ ] Idle time scheduling using WorkManager
- [ ] Automatic execution at 2-3 AM
- [ ] Battery optimization exception handling
- [ ] Progress notification

**Success Criteria**:
- Calendar data collected successfully
- All-inData meaningfully grouped
- Stable background execution
- Reasonable memory usage levels

---

### Phase 5: Settings & Data Management (Week 11)
**Goal**: Settings page and data management functionality

#### Milestones:
- [ ] M5.1: Settings page UI implementation
- [ ] M5.2: Permission management interface
- [ ] M5.3: Data backup/restore functionality
- [ ] M5.4: LLM model management
- [ ] M5.5: Manual learning functionality

#### Detailed To-Do:
**T5.1: Settings UI**
- [ ] Implement SettingsScreen Compose
- [ ] Group settings by category
- [ ] Toggle, button, slider components
- [ ] Access from top-right icon

**T5.2: Permission Management**
- [ ] Display status for each permission
- [ ] Manual permission request functionality
- [ ] Permission explanation dialogs
- [ ] Link to system settings

**T5.3: Data Backup**
- [ ] Generate encrypted backup files
- [ ] External storage access
- [ ] Backup progress display
- [ ] Backup completion notification

**T5.4: Data Restore**
- [ ] Backup file validation
- [ ] User password authentication
- [ ] Data restore process
- [ ] Restart after restoration

**Success Criteria**:
- All settings items function properly
- Backup/restore completes without errors
- User-friendly interface

---

### Phase 6: Security & Encryption (Week 12)
**Goal**: Encryption system and security enhancement

#### Milestones:
- [ ] M6.1: Android KeyStore-based key management
- [ ] M6.2: AES-256 + Argon2 encryption system
- [ ] M6.3: All sensitive data encryption
- [ ] M6.4: App-external file security
- [ ] M6.5: Device change scenario testing

#### Detailed To-Do:
**T6.1: Key Management System**
- [ ] Android KeyStore key generation
- [ ] Hardware security key utilization
- [ ] Key rotation policy implementation
- [ ] Key backup/recovery strategy

**T6.2: Encryption System**
- [ ] AES-256-GCM implementation
- [ ] Argon2 user key derivation
- [ ] PBKDF2 additional security layer
- [ ] Integrity verification system

**T6.3: Data Protection**
- [ ] VoicechatData encryption
- [ ] CalMemoData encryption
- [ ] LLM model file protection
- [ ] All-inData secure storage

**Success Criteria**:
- All sensitive data encrypted in storage
- Files cannot be opened outside the app
- Normal restoration during device change

---

### Phase 7: Performance & Optimization (Week 13)
**Goal**: Performance optimization and stability improvement

#### Milestones:
- [ ] M7.1: Memory usage optimization
- [ ] M7.2: Battery consumption minimization
- [ ] M7.3: Response speed improvement
- [ ] M7.4: App startup time reduction
- [ ] M7.5: Stability testing

#### Detailed To-Do:
**T7.1: Memory Optimization**
- [ ] Memory profiling
- [ ] Memory leak elimination
- [ ] Efficient LLM model loading
- [ ] Cache system implementation

**T7.2: Performance Tuning**
- [ ] CPU usage optimization
- [ ] Hardware acceleration utilization
- [ ] Asynchronous I/O operations
- [ ] Resource usage minimization

**Success Criteria**:
- Memory usage within acceptable range
- App startup time under 5 seconds
- 24-hour continuous operation capability

---

### Phase 8: UI/UX Polish & Final Testing (Week 14-15)
**Goal**: UI improvement and final testing

#### Milestones:
- [ ] M8.1: UI design completeness enhancement
- [ ] M8.2: User experience improvement
- [ ] M8.3: Accessibility feature addition
- [ ] M8.4: Comprehensive testing
- [ ] M8.5: Final APK build

#### Detailed To-Do:
**T8.1: UI Polishing**
- [ ] Complete pastel tone theme
- [ ] Animations and transition effects
- [ ] Icon design improvement
- [ ] Responsive layout optimization

**T8.2: UX Improvement**
- [ ] User onboarding flow
- [ ] Help and tutorials
- [ ] Error handling improvement
- [ ] Feedback system

**T8.3: Testing**
- [ ] Complete unit tests
- [ ] Run integration tests
- [ ] Various actual device testing
- [ ] Performance benchmarks

**Success Criteria**:
- All functions operate stably
- User-friendly interface completion
- Deployment-ready quality achievement

---

## 📈 Success Metrics & Validation Criteria

### Technical Success Metrics
- **LLM Performance**: Response time under 3 seconds, reasonable memory usage
- **Voice Processing**: STT accuracy over 90%, natural TTS quality
- **Stability**: 24-hour continuous operation, crash rate under 1%
- **Usability**: App startup time under 5 seconds, intuitive UI/UX

### Business Success Metrics
- **User Engagement**: Average 30+ minutes continuous conversation
- **Personalization Quality**: User-specific response accuracy
- **Offline Completeness**: All functions work without internet

---

## 🎯 Risk Management per Phase

### Risk Scenarios & Countermeasures
1. **LLM Performance Issues** → Switch to optimized models, enhance quantization
2. **Memory Shortage** → Implement efficient loading, optimize memory usage
3. **Voice Processing Quality Degradation** → Parameter tuning, alternative library review
4. **Development Schedule Delays** → Re-prioritize features, reduce MVP scope

### Phase-by-Phase Validation Points
Must verify following items upon each Phase completion:
- [ ] APK build success
- [ ] Functionality on actual device
- [ ] Memory usage within thresholds
- [ ] User scenario test pass

---

## 📝 Development Progress Management

### Checklist-Based Management
- Track progress with checkboxes for each To-Do item
- Weekly progress review and adjustments
- Monthly Milestone achievement evaluation

### Documentation Rules
- Generate Korean MD file upon each Phase completion
- Record technical issues and solutions
- Maintain performance benchmark results

---

This development plan presents the most realistic and phased approach based on 10 previous failures. The key to success is verifying actual working results at each Phase and progressively building upon them.