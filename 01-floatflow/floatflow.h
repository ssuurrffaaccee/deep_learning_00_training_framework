#pragma once
#include <vector>
#include <queue>
#include <unordered_map>
#include <memory>
#include <string>
#include <functional>
#include <cmath>
//Value with Gradient/Derivative
struct DValue
{
    float v = 0.0f;
    float dv = 0.0f;
    bool needD = true;
};
struct DValuePool
{
    static DValuePool &get()
    {
        static DValuePool instance;
        return instance;
    }
    ~DValuePool()
    {
        for (auto &v : vs_)
        {
            delete v;
        }
    }
    DValue *alloc()
    {
        DValue *z = new DValue{};
        vs_.push_back(z);
        return z;
    }
    std::vector<DValue *> vs_;
};
DValue *alloc()
{
    return DValuePool::get().alloc();
}
// needDerivative inference
DValue *inferenceThenCreate(DValue *x, DValue *y)
{
    DValue *z = alloc();
    z->needD = x->needD || y->needD;
    return z;
}
DValue *inferenceThenCreate(DValue *x)
{
    DValue *z = alloc();
    z->needD = x->needD;
    return z;
}
// Grad Tape
struct Tape
{
    std::deque<std::function<void()>> forwardOps;
    std::deque<std::function<void()>> backwardOps;
    void forward()
    {
        for (auto &f : forwardOps)
        {
            f();
        }
    }
    void backward()
    {
        for (auto &b : backwardOps)
        {
            b();
        }
    }
};
// Values
DValue *parameter()
{
    DValue *z = alloc();
    z->v = 0.0f;
    z->dv = 0.0f;
    z->needD = true;
    return z;
}
DValue *placeholder()
{
    DValue *z = alloc();
    z->v = 0.0f;
    z->dv = 0.0f;
    z->needD = false;
    return z;
}
DValue *constant(float v)
{
    DValue *z = alloc();
    z->v = v;
    z->dv = 0.0f;
    z->needD = false;
    return z;
}
//Base Operators
DValue *add(DValue *x, DValue *y, Tape *t)
{
    DValue *z = inferenceThenCreate(x, y);
    t->forwardOps.push_back([x, y, z]()
                            { z->v = x->v + y->v;  x->dv = 0.0f;
                              y->dv = 0.0f;
                              z->dv = 0.0f; });
    if (z->needD)
    {
        t->backwardOps.push_front([x, y, z]()
                                  {
                if(x->needD){
                   x->dv = z->dv;
                }
                if(y->needD){
                  y ->dv = z->dv;
                } });
    }
    return z;
}
DValue *mul(DValue *x, DValue *y, Tape *t)
{
    DValue *z = inferenceThenCreate(x, y);
    t->forwardOps.push_back([x, y, z]()
                            { z->v = x->v * y->v; 
                              x->dv = 0.0f;
                              y->dv = 0.0f;
                              z->dv = 0.0f; });
    if (z->needD)
    {
        t->backwardOps.push_front([x, y, z]()
                                  { 
                if(x->needD){
                    x->dv +=  y->v * z->dv;
                }
                if(y->needD){
                    y->dv += x->v * z->dv;
                } });
    }
    return z;
}
DValue *minus(DValue *x, Tape *t)
{
    DValue *z = inferenceThenCreate(x);
    t->forwardOps.push_back([x, z]()
                            { z->v = -1.0f * (x->v); 
                            x->dv = 0.0f;
                            z->dv = 0.0f; });
    if (z->needD)
    {
        t->backwardOps.push_front([x, z]()
                                  { 
                if(x->needD){
                    x->dv += -1.0f*z->dv;
                } });
    }
    return z;
}
std::shared_ptr<std::unordered_map<std::string, float>> createReLUState()
{
    auto t = std::make_shared<std::unordered_map<std::string, float>>();
    (*t)["isActive"] = 1.0f;
    return t;
}
DValue *relu(DValue *x, Tape *t)
{
    DValue *z = inferenceThenCreate(x);
    auto sharedState = createReLUState();
    t->forwardOps.push_back([x, z, sharedState]()
                            { if(x->v>0.0f){
                                z->v = x->v;
                                (*sharedState)["isActive"] = 1.0f;
                                x->dv = 0.0f;
                                z->dv = 0.0f;
                                return;
                               }
                               z->v = 0.0f;
                                (*sharedState)["isActive"] = -1.0f; });
    if (z->needD)
    {
        t->backwardOps.push_front([x, z, sharedState]()
                                  { 
                if(x->needD){
                    float isActive = (*sharedState)["isActive"];
                    if(isActive > 0){
                       x->dv += z->dv;
                    }
                } });
    }
    return z;
}
//Compound Operators
DValue *linear(DValue *a, DValue *x, DValue *b, Tape *t)
{
    return add(mul(a, x, t), b, t);
}
DValue *L2(DValue *x, DValue *y, Tape *t)
{
    DValue *z = inferenceThenCreate(x, y);
    t->forwardOps.push_back([x, y, z]()
                            { z->v = (x->v - y->v) * (x->v - y->v); x->dv = 0.0f;
                              y->dv = 0.0f;
                              z->dv = 0.0f; });
    if (z->needD)
    {
        t->backwardOps.push_front([x, y, z]()
                                  { 
                if(x->needD){
                    x->dv +=  2.0f*(x->v - y->v)*z->dv;
                }
                if(y->needD){
                    y->dv +=  2.0f*(y->v-x->v)*z->dv;
                } });
    }
    return z;
}
DValue *L2_(DValue *x, DValue *y, Tape *t)
{
    return mul(add(x, minus(y, t), t), add(x, minus(y, t), t), t);
}
struct ParameterServer
{
    struct Value
    {
        std::string name;
        std::string optiName;
        std::unordered_map<std::string, float> attrs;
        float v = 0.0f;
    };
    ParameterServer()
    {
        addOptis();
    }
    void regist(const Value &v)
    {
        vs_[v.name] = v;
        if (v.optiName == "adagrad")
        {
            auto vAdagrad = v;
            vAdagrad.name = v.name + "_adagrad";
            vAdagrad.v = 0.0f;
            vs_[vAdagrad.name] = vAdagrad;
        }
        if (v.optiName == "adam")
        {
            auto vAdamVel = v;
            vAdamVel.name = v.name + "_adam_vel";
            vAdamVel.v = 0.0f;
            vs_[vAdamVel.name] = vAdamVel;
            auto vAdamMov = v;
            vAdamMov.name = v.name + "_adam_mov";
            vAdamMov.v = 0.0f;
            vs_[vAdamMov.name] = vAdamMov;
        }
    }
    float pull(const std::string &var)
    {
        return vs_[var].v;
    }
    void push(const std::string &var, float dv)
    {
        auto &v = vs_[var];
        auto optiType = v.optiName;
        auto opti = optis_[optiType];
        v.v = opti(v, dv, this);
        return;
    }
    void store(const std::string &var, float v)
    {
        vs_[var].v = v;
        return;
    }
    void addOptis()
    {
        optis_["sgd"] = [](Value &v, float dv, ParameterServer *ps) -> float
        {
            float learningRate = v.attrs["lr"];
            return v.v - learningRate * dv;
        };
        optis_["adagrad"] = [](Value &v, float dv, ParameterServer *ps) -> float
        {
            float learningRate = v.attrs["lr"];
            float h = ps->pull(v.name + "_adagrad");
            h = h + dv * dv;
            ps->store(v.name + "_adagrad", h);
            return v.v - learningRate / (std::sqrt(h) + 1e-7) * dv;
        };
        optis_["adam"] = [](Value &v, float dv, ParameterServer *ps) -> float
        {
            float learningRate = v.attrs["lr"];
            float beta0 = v.attrs["beta0"];
            float beta1 = v.attrs["beta1"];
            float oldVel = ps->pull(v.name + "_adam_vel");
            float oldMov = ps->pull(v.name + "_adam_mov");
            float mov = beta0 * oldMov + (1 - beta0) * dv;
            float vel = beta1 * oldVel + (1 - beta1) * dv * dv;
            ps->store(v.name + "_adam_vel", vel);
            ps->store(v.name + "_adam_mov", mov);
            return v.v - learningRate * mov / (std::sqrt(vel) + 1e-8);
        };
    }
    std::unordered_map<std::string, Value> vs_;
    std::unordered_map<std::string, std::function<float(Value &, float, ParameterServer *)>> optis_;
};