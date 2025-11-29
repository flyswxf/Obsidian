# 酒店预约系统领域模型

下图为核心领域模型，采用 Mermaid 类图表示，包含主要实体、值对象与关系。

```mermaid
classDiagram
direction LR

class Hotel {
  +id: UUID
  +name: string
  +address: string
  +phone: string
}

class RoomType {
  +id: UUID
  +name: string
  +capacity: int
  +basePrice: Money
}

class Room {
  +id: UUID
  +number: string
  +floor: int
  +status: RoomStatus
}

class RatePlan {
  +id: UUID
  +name: string
  +cancellationPolicy: string
  +pricingRule: PricingRule
}

class Amenity {
  +id: UUID
  +name: string
}

class Guest {
  +id: UUID
  +name: string
  +email: string
  +phone: string
}

class Reservation {
  +id: UUID
  +code: string
  +status: ReservationStatus
  +checkIn: Date
  +checkOut: Date
  +totalAmount: Money
}

class ReservationItem {
  +id: UUID
  +qty: int
  +unitPrice: Money
}

class Payment {
  +id: UUID
  +amount: Money
  +method: PaymentMethod
  +status: PaymentStatus
  +paidAt: DateTime
}

class Invoice {
  +id: UUID
  +number: string
  +issuedAt: DateTime
  +amount: Money
}

class Coupon {
  +id: UUID
  +code: string
  +discount: Percentage|Money
  +validUntil: Date
}

class BookingSource {
  +id: UUID
  +channel: ChannelType
}

Hotel "1" o-- "many" Room : 包含
Hotel "1" o-- "many" RatePlan : 提供
Room "1" -- "1" RoomType : 类型
Room "many" o-- "many" Amenity : 配备

Reservation "1" o-- "many" ReservationItem : 包含
ReservationItem "1" -- "1" Room : 对应房间
Reservation "1" -- "1" Guest : 由旅客预订
Reservation "1" -- "1" RatePlan : 应用价格计划
Reservation "0..1" o-- "many" Coupon : 使用
Reservation "0..1" -- "many" Payment : 支付
Reservation "0..1" -- "1" Invoice : 开具发票
Reservation "1" -- "1" BookingSource : 渠道

class Money {
  +currency: string
  +amount: decimal
}

class PricingRule {
  +calc(price: Money, dateRange: DateRange): Money
}

class DateRange {
  +start: Date
  +end: Date
}

class RoomStatus {
  <<enumeration>>
  Available
  Occupied
  OutOfService
}

class ReservationStatus {
  <<enumeration>>
  Pending
  Confirmed
  CheckedIn
  CheckedOut
  Cancelled
}

class PaymentMethod {
  <<enumeration>>
  CreditCard
  PayPal
  BankTransfer
  Cash
}

class PaymentStatus {
  <<enumeration>>
  Initiated
  Succeeded
  Failed
  Refunded
}

class ChannelType {
  <<enumeration>>
  Web
  Phone
  OTA
}
```

关键约束与说明：
- 一次预约 `Reservation` 可以包含多间房，通过 `ReservationItem` 关联具体 `Room` 与数量。
- 定价由 `RatePlan` 与 `PricingRule` 决定，可叠加 `Coupon` 折扣，最终生成 `Invoice` 并记录 `Payment`。
- 可用性依赖 `Room.status` 与预约时间区间 `DateRange`，取消规则在 `RatePlan.cancellationPolicy`。
- 渠道来源 `BookingSource` 区分官网、电话及第三方 OTA。

---

# 预定机票用例系统顺序图

## 用例描述
**用例名：** 预订机票  
**主要参与者：** 订票人，第三方支付平台  
**简要描述：** 乘客完成从查询到支付的购票流程，系统生成有效电子票并关联乘客与航班。

## 系统顺序图 (SSD)

下图展示了预定机票用例的主要事件流，包括乘客与系统的交互以及系统与第三方支付平台的交互。

```mermaid
sequenceDiagram
    participant P as 乘客
    participant S as 机票预订系统
    participant PS as 第三方支付平台

    Note over P,PS: 主事件流
    
    P->>S: 1. 输入查询条件(出发地, 目的地, 日期, 人数)
    S->>S: 2. 调用航班信息服务
    S-->>P: 返回可用航班列表(价格, 舱位, 退改规则)
    
    P->>S: 3. 选择航班和舱位, 填写乘客信息
    
    alt 座位充足
        S->>S: 4. 锁定座位
        S->>S: 5. 生成待支付机票
        S-->>P: 显示订单摘要与总价
        
        P->>S: 6. 确认并发起支付
        S->>PS: 7. 调用支付接口(订单信息, 金额)
        PS-->>S: 返回支付结果
        
        alt 支付成功
            S-->>P: 8. 显示订票成功界面
            Note over P,S: 9. 用例成功结束
        else 支付失败/取消
            S->>S: 7a.1 保留机票占有一定时长
            S-->>P: 显示支付失败信息
            Note over P,S: 7a.2 用例失败结束(超时自动释放座位)
        end
        
    else 座位不足
        S-->>P: 3a.1 提示无票并建议改选舱位或日期
        Note over P,S: 3a.2 用例失败结束
    end
```

## 关键系统事件说明

1. **输入查询条件** - 乘客向系统提供搜索参数
2. **返回航班列表** - 系统响应查询并返回可用选项
3. **选择航班和填写信息** - 乘客做出选择并提供必要信息
4. **锁定座位** - 系统预留选定座位
5. **生成订单** - 系统创建待支付的机票订单
6. **发起支付** - 乘客确认订单并启动支付流程
7. **处理支付** - 系统与第三方支付平台交互
8. **显示结果** - 系统向乘客反馈最终状态

## 异常流程处理

- **座位不足 (3a)：** 系统检测到库存不足时，提供替代建议
- **支付失败 (7a)：** 系统实施座位保留机制，防止立即释放，给用户重试机会