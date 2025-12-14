#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <algorithm>
#include <iomanip>
#include <string>

// Abstract Strategy Interface
class DiscountStrategy
{
public:
    virtual ~DiscountStrategy() = default;
    virtual double applyDiscount(double originalPrice, int quantity) = 0;
};

// 1. Flat Rate Strategy
class FlatRateStrategy : public DiscountStrategy
{
private:
    double discountRate;

public:
    FlatRateStrategy(double rate) : discountRate(rate) {}

    void setDiscountRate(double rate)
    {
        discountRate = rate;
    }

    double applyDiscount(double originalPrice, int quantity) override
    {
        return originalPrice * discountRate;
    }
};

// 2. Threshold Strategy (e.g., Spend 1000 get 300 off)
class ThresholdStrategy : public DiscountStrategy
{
private:
    double threshold;
    double reduction;

public:
    ThresholdStrategy(double thresholdVal, double reductionVal)
        : threshold(thresholdVal), reduction(reductionVal) {}

    void setThreshold(double val) { threshold = val; }
    void setReduction(double val) { reduction = val; }

    double applyDiscount(double originalPrice, int quantity) override
    {
        if (originalPrice >= threshold)
        {
            return originalPrice - reduction;
        }
        return originalPrice;
    }
};

// 3. Quantity Tiered Strategy
class QuantityTieredStrategy : public DiscountStrategy
{
private:
    // map automatically sorts keys
    std::map<int, double> tierMap;

public:
    void addTier(int minQuantity, double discountRate)
    {
        tierMap[minQuantity] = discountRate;
    }

    void clearTiers()
    {
        tierMap.clear();
    }

    double applyDiscount(double originalPrice, int quantity) override
    {
        double rate = 1.0;
        // Find the largest key <= quantity
        for (auto it = tierMap.rbegin(); it != tierMap.rend(); ++it)
        {
            if (quantity >= it->first)
            {
                rate = it->second;
                break;
            }
        }
        return originalPrice * rate;
    }
};

// Composite Strategy
class CompositeDiscountStrategy : public DiscountStrategy
{
private:
    std::vector<std::shared_ptr<DiscountStrategy>> strategies;

public:
    void addStrategy(std::shared_ptr<DiscountStrategy> strategy)
    {
        strategies.push_back(strategy);
    }

    void removeStrategy(std::shared_ptr<DiscountStrategy> strategy)
    {
        auto it = std::remove(strategies.begin(), strategies.end(), strategy);
        strategies.erase(it, strategies.end());
    }

    void clearStrategies()
    {
        strategies.clear();
    }

    double applyDiscount(double originalPrice, int quantity) override
    {
        double currentPrice = originalPrice;
        for (const auto &strategy : strategies)
        {
            // double oldPrice = currentPrice;
            currentPrice = strategy->applyDiscount(currentPrice, quantity);
        }
        return currentPrice;
    }
};

// Helper function to print scenarios
void printScenario(const std::string &description, double originalPrice, int quantity, std::shared_ptr<DiscountStrategy> strategy)
{
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Scenario: " << description << std::endl;
    std::cout << "Original Price: " << originalPrice << ", Quantity: " << quantity << std::endl;
    double finalPrice = strategy->applyDiscount(originalPrice, quantity);
    std::cout << "Final Price: " << std::fixed << std::setprecision(2) << finalPrice << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
}

int main()
{
    // Initialize strategies
    auto flatRate = std::make_shared<FlatRateStrategy>(0.88); // 88% off

    auto threshold = std::make_shared<ThresholdStrategy>(1000.0, 300.0); // Spend 1000 get 300 off

    auto quantityTiered = std::make_shared<QuantityTieredStrategy>();
    quantityTiered->addTier(1, 0.90); // 1 item 90%
    quantityTiered->addTier(2, 0.80); // 2 items 80%
    quantityTiered->addTier(3, 0.75); // 3+ items 75%

    // 1. Single Strategy: Flat Rate
    printScenario("Single - Flat Rate 88%", 2000, 1, flatRate);

    // 2. Single Strategy: Threshold
    printScenario("Single - Spend 1000 get 300 off", 1200, 1, threshold);

    // 3. Single Strategy: Quantity Tiered (Buy 3)
    // Total price 3000 (Unit price 1000 * 3)
    printScenario("Single - Quantity Tiered (5 items -> 75%)", 3000, 5, quantityTiered);

    // 4. Composite Strategy: Quantity -> Threshold -> Flat Rate
    auto composite = std::make_shared<CompositeDiscountStrategy>();
    composite->addStrategy(quantityTiered);
    composite->addStrategy(threshold);
    composite->addStrategy(flatRate);

    std::cout << "\n=== Composite Strategy Test ===" << std::endl;
    // 3 items, Total 3000
    // Expected:
    // 1. Quantity(0.75): 3000 * 0.75 = 2250
    // 2. Threshold(-300): 2250 - 300 = 1950
    // 3. Flat(0.88): 1950 * 0.88 = 1716
    printScenario("Composite (Quantity -> Threshold -> Flat)", 3000, 3, composite);

    // 5. Dynamic Parameter Modification
    std::cout << "\n=== Dynamic Parameter Modification Test ===" << std::endl;
    std::cout << ">> Changing Threshold Strategy to: Spend 1000 get 600 off" << std::endl;
    threshold->setReduction(600.0);
    // Expected:
    // 1. Quantity(0.75): 2250
    // 2. Threshold(-600): 2250 - 600 = 1650
    // 3. Flat(0.88): 1650 * 0.88 = 1452
    printScenario("Composite after parameter change", 3000, 3, composite);

    // 6. Dynamic Structure Modification
    std::cout << "\n=== Dynamic Structure Modification Test ===" << std::endl;
    std::cout << ">> Removing Flat Rate Strategy" << std::endl;
    composite->removeStrategy(flatRate);
    // Expected:
    // 1. Quantity(0.75): 2250
    // 2. Threshold(-600): 1650
    printScenario("Composite after removing Flat Rate", 3000, 3, composite);

    return 0;
}
