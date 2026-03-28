#include <iostream>
#include <string>
#include <memory>
#include <map>
#include <queue>
#include <utility>

enum class Weather
{
    HOT,
    COLD
};

// --- FlyBehavior Strategy ---
struct FlyBehavior
{
    virtual ~FlyBehavior() = default;
    virtual std::string fly() const = 0;
};

struct FlyWithWings : FlyBehavior
{
    std::string fly() const override { return "FlyWithWings"; }
};
struct FlyWithPropulsion : FlyBehavior
{
    std::string fly() const override { return "FlyWithPropulsion"; }
};
struct FlyWithPropeller : FlyBehavior
{
    std::string fly() const override { return "FlyWithPropeller"; }
};
struct FlyNoWay : FlyBehavior
{
    std::string fly() const override { return "FlyNoWay"; }
};

// --- QuackBehavior Strategy ---
struct QuackBehavior
{
    virtual ~QuackBehavior() = default;
    virtual std::string quack() const = 0;
};

struct Quack : QuackBehavior
{
    std::string quack() const override { return "Quack"; }
};
struct Squick : QuackBehavior
{
    std::string quack() const override { return "Squick"; }
};
struct MuteQuack : QuackBehavior
{
    std::string quack() const override { return "MuteQuack"; }
};

// --- Clothing Strategy ---
struct Clothing
{
    virtual ~Clothing() = default;
    virtual std::string wear() const = 0;
};

struct BoyWinterClothing : Clothing
{
    std::string wear() const override { return "Boy Winter Coat"; }
};
struct GirlWinterClothing : Clothing
{
    std::string wear() const override { return "Girl Winter Coat"; }
};
struct AdultSummerClothing : Clothing
{
    std::string wear() const override { return "Adult Summer Shirt"; }
};

// --- Duck Hierarchy ---
class Duck
{
public:
    explicit Duck(std::string n) : name(std::move(n)), isAlive(true) {}
    virtual ~Duck() = default;

    std::string performFly() const { return name + "." + (flyBehavior ? flyBehavior->fly() : std::string("")); }
    std::string performSwim() const { return name + ".Swim"; }
    std::string performQuack() const { return name + "." + (quackBehavior ? quackBehavior->quack() : std::string("")); }
    virtual std::string display() const { return name; }
    void setFlyBehavior(std::unique_ptr<FlyBehavior> fb) { flyBehavior = std::move(fb); }
    void setQuackBehavior(std::unique_ptr<QuackBehavior> qb) { quackBehavior = std::move(qb); }
    std::string die()
    {
        isAlive = false;
        return name + ".Dead";
    }
    const std::string &getName() const { return name; }

protected:
    std::unique_ptr<FlyBehavior> flyBehavior;
    std::unique_ptr<QuackBehavior> quackBehavior;
    bool isAlive;
    std::string name;
};

class MallardDuck : public Duck
{
public:
    explicit MallardDuck(const std::string &n) : Duck(n)
    {
        setFlyBehavior(std::make_unique<FlyWithWings>());
        setQuackBehavior(std::make_unique<Quack>());
    }
    std::string display() const override { return "MallardDuck"; }
};

class RedHeadDuck : public Duck
{
public:
    explicit RedHeadDuck(const std::string &n) : Duck(n)
    {
        setFlyBehavior(std::make_unique<FlyWithWings>());
        setQuackBehavior(std::make_unique<Quack>());
    }
    std::string display() const override { return "RedHeadDuck"; }
};

class RubberDuck : public Duck
{
public:
    explicit RubberDuck(const std::string &n) : Duck(n)
    {
        setFlyBehavior(std::make_unique<FlyNoWay>());
        setQuackBehavior(std::make_unique<Squick>());
    }
    std::string display() const override { return "RubberDuck"; }
};

class DecoyDuck : public Duck
{
public:
    explicit DecoyDuck(const std::string &n) : Duck(n)
    {
        setFlyBehavior(std::make_unique<FlyNoWay>());
        setQuackBehavior(std::make_unique<MuteQuack>());
    }
    std::string display() const override { return "DecoyDuck"; }
};

// --- Person Hierarchy ---
class Person
{
public:
    explicit Person(std::string n) : name(std::move(n)) {}
    virtual ~Person() = default;
    std::string performWalk() const { return name + ".Walk"; }
    virtual std::string display() const { return name + " wears " + (clothing ? clothing->wear() : "nothing"); }
    void setClothing(std::unique_ptr<Clothing> c) { clothing = std::move(c); }
    const std::string &getName() const { return name; }

protected:
    std::unique_ptr<Clothing> clothing;
    std::string name;
};

class Hunter : public Person
{
public:
    explicit Hunter(const std::string &n) : Person(n)
    {
        setClothing(std::make_unique<AdultSummerClothing>());
    }
    std::string shoot(Duck &target)
    {
        return name + ".Shoots " + target.getName();
    }
    std::string display() const override { return std::string("Hunter") + " wears " + (clothing ? clothing->wear() : std::string("nothing")); }
};

class Boy : public Person
{
public:
    explicit Boy(const std::string &n) : Person(n)
    {
        setClothing(std::make_unique<BoyWinterClothing>());
    }
    std::string display() const override { return std::string("Boy") + " wears " + (clothing ? clothing->wear() : std::string("nothing")); }
};
class Girl : public Person
{
public:
    explicit Girl(const std::string &n) : Person(n)
    {
        setClothing(std::make_unique<GirlWinterClothing>());
    }
    std::string display() const override { return std::string("Girl") + " wears " + (clothing ? clothing->wear() : std::string("nothing")); }
};

// --- Aircraft Hierarchy ---
class Aircraft
{
public:
    explicit Aircraft(std::string n) : name(std::move(n)) { setFlyBehavior(std::make_unique<FlyNoWay>()); }
    virtual ~Aircraft() = default;
    std::string performFly() const { return name + "." + (flyBehavior ? flyBehavior->fly() : std::string("")); }
    std::string takeOff() { return name + ".TakeOff"; }
    virtual std::string display() const { return name; }
    void setFlyBehavior(std::unique_ptr<FlyBehavior> fb) { flyBehavior = std::move(fb); }
    const std::string &getName() const { return name; }

protected:
    std::unique_ptr<FlyBehavior> flyBehavior;
    std::string name;
};

class Boeing : public Aircraft
{
public:
    explicit Boeing(const std::string &n) : Aircraft(n)
    {
        setFlyBehavior(std::make_unique<FlyWithPropulsion>());
    }
    std::string display() const override { return "Boeing"; }
};
class Apache : public Aircraft
{
public:
    explicit Apache(const std::string &n) : Aircraft(n)
    {
        setFlyBehavior(std::make_unique<FlyWithPropeller>());
    }
    std::string display() const override { return "Apache"; }
};

// --- Scene Logic ---
std::map<std::string, std::queue<std::string>> produceScene(Weather weather)
{
    std::map<std::string, std::queue<std::string>> description;

    if (weather == Weather::HOT)
    {
        std::queue<std::string> scene;

        Hunter hunter("Hunter");
        scene.push(hunter.display());
        scene.push(hunter.performWalk());

        MallardDuck mallard("MallardDuck");
        scene.push(mallard.display());
        scene.push(mallard.performSwim());

        RedHeadDuck red("RedHeadDuck");
        scene.push(red.display());
        scene.push(red.performSwim());

        RubberDuck rubber("RubberDuck");
        scene.push(rubber.display());
        scene.push(rubber.performSwim());

        scene.push("A few seconds later...");
        scene.push(mallard.performFly());

        scene.push("A few seconds later...");
        scene.push(hunter.shoot(mallard));
        scene.push(mallard.die());

        description["Scene1:Hot"] = scene;
    }
    else
    {
        std::queue<std::string> scene;

        Boy boy("Boy");
        scene.push(boy.display());
        scene.push(boy.performWalk());

        Girl girl("Girl");
        scene.push(girl.display());
        scene.push(girl.performWalk());

        Boeing boeing("Boeing");
        scene.push(boeing.display());
        Apache apache("Apache");
        scene.push(apache.display());

        scene.push(boeing.takeOff());
        scene.push(apache.takeOff());
        scene.push(boeing.performFly());
        scene.push(apache.performFly());

        description["Scene2:Cold"] = scene;
    }

    return description;
}

void produceAnimate(const std::string &file, const std::map<std::string, std::queue<std::string>> &description)
{
    for (const auto &kv : description)
    {
        std::cout << kv.first << '\n';
        auto q = kv.second;
        while (!q.empty())
        {
            std::cout << q.front() << '\n';
            q.pop();
        }
        std::cout << "---" << '\n';
    }
}

class SimulationController
{
public:
    void setWeather(Weather w) { weather = w; }
    std::map<std::string, std::queue<std::string>> runScene() { return produceScene(weather); }

private:
    Weather weather{Weather::HOT};
};

int main()
{
    SimulationController controller;
    controller.setWeather(Weather::HOT);
    auto hotDesc = controller.runScene();
    produceAnimate("DuckHunting_Hot.json", hotDesc);

    controller.setWeather(Weather::COLD);
    auto coldDesc = controller.runScene();
    produceAnimate("DuckHunting_Cold.json", coldDesc);
    return 0;
}
