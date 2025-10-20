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