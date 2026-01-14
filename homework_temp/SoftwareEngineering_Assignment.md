# 软件工程作业

## 1. 领域模型 (Domain Model) (10 分)

**概念:**
领域模型捕捉了在线教育平台环境中的重要概念类、它们的属性以及它们之间的关联。

**识别出的概念类:**
- **User (用户)** (抽象类), **Student (学生)**, **Teacher (教师)**, **Administrator (管理员)**
- **Course (课程)** (抽象类), **VideoCourse (视频课程)**, **GraphicCourse (图文课程)**, **LiveBroadcastCourse (直播课程)**
- **CourseRegistration (课程注册)**, **WaitlistEntry (候补名单记录)**
- **Payment (支付)**
- **LearningReport (学习报告)**, **LearningProgress (学习进度)**

**Mermaid 领域模型图:**

```mermaid
classDiagram
    class User {
        name (姓名)
        email (邮箱)
        password (密码)
    }
    class Student {
        age (年龄)
        learningRequirements (学习需求)
    }
    class Teacher {
        specialization (专业领域)
    }
    class Administrator {
    }
    class Course {
        title (标题)
        description (描述)
        category (类别)
        difficultyLevel (难度等级)
        studyDuration (学习时长)
        price (价格)
        capacity (容量)
        status (状态)
    }
    class VideoCourse {
        videoUrl (视频链接)
    }
    class GraphicCourse {
        content (内容)
    }
    class LiveBroadcastCourse {
        schedule (时间表)
        streamUrl (直播流链接)
    }
    class CourseRegistration {
        registrationDate (注册日期)
        status (状态)
    }
    class WaitlistEntry {
        joinDate (加入日期)
    }
    class Payment {
        amount (金额)
        paymentMethod (支付方式)
        transactionId (交易ID)
        status (状态)
    }
    class LearningProgress {
        viewingDuration (观看时长)
        interactionCount (互动次数)
        lastAccessDate (最后访问日期)
    }
    class LearningReport {
        generatedDate (生成日期)
        content (内容)
    }

    User <|-- Student
    User <|-- Teacher
    User <|-- Administrator
    
    Course <|-- VideoCourse
    Course <|-- GraphicCourse
    Course <|-- LiveBroadcastCourse

    Teacher "1" --> "*" Course : manages (管理)
    Student "1" --> "*" CourseRegistration : has (拥有)
    CourseRegistration "*" --> "1" Course : for (针对)
    Student "1" --> "*" WaitlistEntry : joins (加入)
    WaitlistEntry "*" --> "1" Course : for (针对)
    
    CourseRegistration "1" --> "0..1" Payment : initiates (发起)
    
    Student "1" --> "*" LearningProgress : tracks (追踪)
    LearningProgress "*" --> "1" Course : in (在...中)
    
    Student "1" --> "*" LearningReport : receives (收到)
```

---

## 2. 用例模型 (Use Case Model) (15 分)

**用例图:**

```mermaid
graph LR
    %% Actors
    Student[学生]
    Teacher[教师]
    Administrator[管理员]
    System[系统]
    Bank[支付系统]

    %% Use Cases
    UC1((创建课程))
    UC2((编辑课程))
    UC3((筛选/查找课程))
    UC4((注册课程))
    UC5((加入候补名单))
    UC6((进行支付))
    UC7((查看进度))
    UC8((管理注册))
    UC9((生成学习报告))

    %% Relationships
    Teacher --> UC1
    Teacher --> UC2
    Teacher --> UC8

    Student --> UC3
    Student --> UC4
    Student --> UC7

    UC4 -. include .-> UC6
    UC4 -. extend .-> UC5
    
    UC6 --> Bank
    
    Administrator --> UC8
    
    System --> UC9
```

**用例描述: 注册课程 (Register for Course)**

| 字段        | 描述                                                                                                                                                                                                                                                                                                                                |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **用例名称**  | 注册课程                                                                                                                                                                                                                                                                                                                              |
| **主要参与者** | 学生                                                                                                                                                                                                                                                                                                                                |
| **利益相关者** | 学生, 教师, 支付系统                                                                                                                                                                                                                                                                                                                      |
| **前置条件**  | 学生已登录。课程存在且开放注册。                                                                                                                                                                                                                                                                                                                  |
| **后置条件**  | 学生已注册该课程，或已被加入候补名单。                                                                                                                                                                                                                                                                                                               |
| **主成功场景** | 1. 学生选择一门特定课程进行注册。<br>2. 系统验证课程可用性且名额未满。<br>3. 系统确定该课程为免费课程。<br>4. 系统将学生注册到该课程。<br>5. 系统创建课程注册记录。<br>6. 系统通知学生注册成功。                                                                                                                                                                                                               |
| **备选场景**  | **2a. 课程已满:**<br>&nbsp;&nbsp;1. 系统通知学生名额已满。<br>&nbsp;&nbsp;2. 学生选择加入候补名单。<br>&nbsp;&nbsp;3. 系统将学生加入候补名单。<br>&nbsp;&nbsp;4. 用例结束。<br><br>**3a. 课程为付费课程:**<br>&nbsp;&nbsp;1. 系统提示进行支付 (微信支付, 支付宝, 银行卡)。<br>&nbsp;&nbsp;2. 学生提供支付详情。<br>&nbsp;&nbsp;3. 系统通过外部支付系统验证支付。<br>&nbsp;&nbsp;4. 支付成功。<br>&nbsp;&nbsp;5. 系统注册学生 (回到主场景步骤 4)。 |

---

## 3. SSD 和操作契约 (SSD and Operation Contract) (10 分)

**系统顺序图 (SSD) - "注册课程":**

```mermaid
sequenceDiagram
    actor Student as 学生
    participant System as 系统

    Student->>System: selectCourse(courseID) (选择课程)
    alt is full (已满)
        System-->>Student: notifyFullAndAskWaitlist() (通知已满并询问候补)
        opt joins waitlist (加入候补)
            Student->>System: joinWaitlist(courseID) (加入候补名单)
            System-->>Student: waitlistConfirmed() (候补确认)
        end
    else has spots (有名额)
        alt is paid (付费)
            System-->>Student: requestPayment(amount) (请求支付)
            Student->>System: makePayment(paymentDetails) (进行支付)
            System-->>Student: paymentSuccess() (支付成功)
        end
        System-->>Student: registrationConfirmed() (注册确认)
    end
```

**操作契约: `registerStudent`**

*注意: 识别系统操作 `registerStudent`，该操作概念上发生在支付后或直接选择后。*

| 字段       | 内容                                                                                                                                                                           |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **操作**   | `registerStudent(studentID, courseID)`                                                                                                                                       |
| **交叉引用** | 用例: 注册课程                                                                                                                                                                     |
| **前置条件** | - `Student` (学生) 存在。<br>- `Course` (课程) 存在。<br>- `Student` 尚未注册该 `Course`。                                                                                                   |
| **后置条件** | - 创建了一个新的 `CourseRegistration` (课程注册) 实例 `r`。<br>- `r.student` 被关联到 `Student`。<br>- `r.course` 被关联到 `Course`。<br>- `r.status` 变为 "Registered" (已注册)。<br>- `r.date` 被设置为当前日期。 |

---

## 4. GRASP 模式 (GRASP Patterns) (15 分)

**场景:** 学生注册课程 (为简化图表，忽略支付环节，专注于对象创建和职责分配)。

**使用的模式:**
1.  **控制器 (Controller):** `RegistrationController` 处理系统事件。
2.  **信息专家 (Information Expert):** `Course` 知道自己是否已满。
3.  **创建者 (Creator):** `Course` 创建 `CourseRegistration`，因为它聚合了注册记录 (或者 `Student` 也可以，但通常 `Course` 管理其名单是常见的领域选择)。
4.  **低耦合 (Low Coupling):** 使用控制器防止 UI 直接与领域对象耦合。

**交互图:**

```mermaid
sequenceDiagram
    participant UI
    participant RC as :RegistrationController
    participant Course as c:Course
    participant Student as s:Student

    Note over UI, RC: 控制器模式 (Controller Pattern)
    UI->>RC: registerStudent(studentID, courseID)
    
    Note over RC, Course: 领域模型检索 (简化)
    RC->>Course: c = findCourse(courseID)
    RC->>Student: s = findStudent(studentID)

    Note over RC, Course: 信息专家 (isFull)
    RC->>Course: isFull()
    activate Course
    Course-->>RC: false
    deactivate Course

    Note over RC, Course: 创建者模式 (Creator Pattern)
    RC->>Course: register(s)
    activate Course
    create participant Reg as r:CourseRegistration
    Course->>Reg: new(s, c)
    Course->>Course: addRegistration(r)
    deactivate Course

    RC-->>UI: success
```

---

## 5. GoF 模式 (GoF Patterns) (20 分)

**需求:**
系统需要根据业务场景灵活切换不同的推荐算法 (基于内容、协同过滤、深度学习)。

**选择的模式:** **策略模式 (Strategy Pattern)**
- **上下文 (Context):** `RecommendationService` (或 `CourseRecommender`)
- **策略接口 (Strategy Interface):** `RecommendationStrategy`
- **具体策略 (Concrete Strategies):** `ContentBasedStrategy`, `CollaborativeFilteringStrategy`, `DeepLearningStrategy`

**类图:**

```mermaid
classDiagram
    class CourseRecommender {
        -strategy: RecommendationStrategy
        +setStrategy(RecommendationStrategy)
        +recommendCourses(studentId): List~Course~
    }

    class RecommendationStrategy {
        <<interface>>
        +recommend(studentId): List~Course~
    }

    class ContentBasedStrategy {
        +recommend(studentId): List~Course~
    }

    class CollaborativeFilteringStrategy {
        +recommend(studentId): List~Course~
    }

    class DeepLearningStrategy {
        +recommend(studentId): List~Course~
    }

    CourseRecommender o--> "1" RecommendationStrategy : uses (使用)
    RecommendationStrategy <|.. ContentBasedStrategy : implements (实现)
    RecommendationStrategy <|.. CollaborativeFilteringStrategy : implements (实现)
    RecommendationStrategy <|.. DeepLearningStrategy : implements (实现)
```

**解释:**
**策略模式** 在这里非常理想，因为它定义了一族算法 (推荐逻辑)，分别封装起来，并使它们可以互换。`CourseRecommender` (上下文) 不需要知道推荐是如何生成的具体细节。它只需要持有 `RecommendationStrategy` 接口的引用。这允许系统在运行时 (例如通过配置或用户偏好) 更改推荐算法，而无需修改客户端代码，符合开闭原则 (Open/Closed Principle)。
