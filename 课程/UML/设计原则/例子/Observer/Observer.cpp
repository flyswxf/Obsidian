#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// 抽象观察者
class Observer {
public:
    virtual void update(int value) = 0;
    virtual ~Observer() {} // 虚析构函数，防止内存泄漏
};

// 具体观察者: 除法
class DivObserver : public Observer {
    int m_div;
public:
    DivObserver(int div) : m_div(div) {}
    void update(int val) override {
        if (m_div != 0)
            cout << val << " div " << m_div << " is " << val / m_div << endl;
    }
};

// 具体观察者: 取模
class ModObserver : public Observer {
    int m_mod;
public:
    ModObserver(int mod) : m_mod(mod) {}
    void update(int val) override {
        if (m_mod != 0)
            cout << val << " mod " << m_mod << " is " << val % m_mod << endl;
    }
};

class Subject {
    int m_value;
    vector<Observer*> m_views; 
public:
    void attach(Observer* obs) {
        m_views.push_back(obs);
    }
    
    void set_value(int value) {
        m_value = value;
        notify();
    }
    
    void notify() {
        for (size_t i = 0; i < m_views.size(); ++i) {
            m_views[i]->update(m_value);
        }
    }
};

int main() {
    Subject subj;
    
    // 创建观察者
    DivObserver divObs(4);
    ModObserver modObs(3);

    // 注册观察者
    subj.attach(&divObs);
    subj.attach(&modObs);

    // 改变状态，触发通知
    subj.set_value(14);
    
    return 0;
}