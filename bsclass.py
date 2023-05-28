import torch
from multipledispatch import dispatch
from torch import nn
from torch.functional import F

normal = torch.distributions.Normal(0,1)

N = normal.cdf

logNp = normal.log_prob

def cpdf(torchobject):
    return torch.exp(logNp(torchobject))


class OptionClass:
    
    def __init__(self, strike, sigma, rf, maturity):
        self.strike, self.sigma, self.rf, self.maturity =  strike, sigma, rf, maturity

    def changesigma(self, newsigma):
        self.sigma = newsigma
        return 0
    
    @dispatch(object, object)
    def getcall(self, s0, time_t):
        dt = time_t
        d1 = (torch.log(s0/self.strike) + (self.rf + self.sigma**2/2) * dt)/(self.sigma * torch.sqrt(dt))
        d2 = d1 - self.sigma * torch.sqrt(dt)
        return s0 * N(d1) - self.strike * torch.exp(-self.rf*dt) * N(d2)

    @dispatch(object, object, object)
    def getcall(self, s0, time_t, vol):
        dt = time_t
        d1 = (torch.log(s0/self.strike) + (self.rf + vol**2/2) * dt)/(vol * torch.sqrt(dt))
        d2 = d1 - vol * torch.sqrt(dt)
        return s0 * N(d1) - self.strike * torch.exp(-self.rf*dt) * N(d2)

    @dispatch(object, object)
    def getput(self, s0, time_t):
        dt = time_t
        d1 = (torch.log(s0/self.strike) + (self.rf + self.sigma**2/2) * dt)/(self.sigma * torch.sqrt(dt))
        d2 = d1 - self.sigma * torch.sqrt(dt)
        return self.strike * torch.exp(-self.rf*dt) * N(-d2) - s0 * N(-d1)
    
    @dispatch(object, object, object)
    def getput(self, s0, time_t, vol):
        dt = time_t
        d1 = (torch.log(s0/self.strike) + (self.rf + vol**2/2) * dt)/(vol * torch.sqrt(dt))
        d2 = d1 - vol * torch.sqrt(dt)
        return self.strike * torch.exp(-self.rf*dt) * N(-d2) - s0 * N(-d1)

    def getdeltacall(self, s0, time_t):
        dt = time_t
        d1 = (torch.log(s0/self.strike) + (self.rf + self.sigma**2/2) * dt)/(self.sigma * torch.sqrt(dt))
        return N(d1)
    
    def getdeltaput(self, s0, time_t):
        dt = time_t
        d1 = (torch.log(s0/self.strike) + (self.rf + self.sigma**2/2) * dt)/(self.sigma * torch.sqrt(dt))
        return N(d1) - 1
    
    def getgamma(self, s0, time_t):
        dt = time_t
        d1 = (torch.log(s0/self.strike) + (self.rf + self.sigma**2/2) * dt)/(self.sigma * torch.sqrt(dt))
        return cpdf(d1) / (s0 * self.sigma * torch.sqrt(dt))
    
    @dispatch(object, object)
    def getvega(self, s0, time_t):
        dt = time_t
        d1 = (torch.log(s0/self.strike) + (self.rf + self.sigma**2/2) * dt)/(self.sigma * torch.sqrt(dt))
        return s0 * cpdf(d1) * torch.sqrt(dt)
    
    @dispatch(object, object, object)
    def getvega(self, s0, time_t, vol):
        dt = time_t
        d1 = (torch.log(s0/self.strike) + (self.rf + vol**2/2) * dt)/(vol * torch.sqrt(dt))
        return s0 * cpdf(d1) * torch.sqrt(dt)
    
    def getvega_iv(self, s0, time_t, vol):
        dt = time_t
        d1 = (torch.log(s0/self.strike) + (self.rf + vol**2/2) * dt)/(vol * torch.sqrt(dt))
        return s0 * cpdf(d1) * torch.sqrt(dt)

    
    def getthetacall(self, s0, time_t):
        dt = time_t
        d1 = (torch.log(s0/self.strike) + (self.rf + self.sigma**2/2) * dt)/(self.sigma * torch.sqrt(dt))
        d2 = d1 - self.sigma * torch.sqrt(dt)
        return -s0*cpdf(d1)*self.sigma / (2*torch.sqrt(dt)) - self.rf*strike*torch.exp(-self.rf*dt)*N(d2)
    
    def getthetaput(self, s0, time_t):
        dt = time_t
        d1 = (torch.log(s0/self.strike) + (self.rf + self.sigma**2/2) * dt)/(self.sigma * torch.sqrt(dt))
        d2 = d1 - self.sigma * torch.sqrt(dt)
        return -s0*cpdf(d1)*self.sigma / (2*torch.sqrt(dt)) + self.rf*strike*torch.exp(-self.rf*dt)*N(d2)
    
    def getrhocall(self, s0, time_t):
        dt = time_t
        d1 = (torch.log(s0/self.strike) + (self.rf + self.sigma**2/2) * dt)/(self.sigma * torch.sqrt(dt))
        d2 = d1 - self.sigma * torch.sqrt(dt)
        return strike*dt*torch.exp(-self.rf*dt)*N(d2)
        
    def getrhoput(self, s0, time_t):
        dt = time_t
        d1 = (torch.log(s0/self.strike) + (self.rf + self.sigma**2/2) * dt)/(self.sigma * torch.sqrt(dt))
        d2 = d1 - self.sigma * torch.sqrt(dt)
        return -strike*dt*torch.exp(-self.rf*dt)*N(-d2)

    def getvanna(self, s0, time_t):
        dt = time_t
        d1 = (torch.log(s0/self.strike) + (self.rf + self.sigma**2/2) * dt)/(self.sigma * torch.sqrt(dt))
        d2 = d1 - self.sigma * torch.sqrt(dt)
        return -torch.exp(-self.rf*dt)*cpdf(d1)*d2/self.sigma
    
    def getvolga(self, s0, time_t): # оно же vomma
        dt = time_t
        d1 = (torch.log(s0/self.strike) + (self.rf + self.sigma**2/2) * dt)/(self.sigma * torch.sqrt(dt))
        d2 = d1 - self.sigma * torch.sqrt(dt)
        return self.getvega(s0, time_t)*d1*d2/self.sigma
    
    def grad_iv(self, V_mkt, s0, time_t, vol, is_call):
        if is_call == True:
            return V_mkt - self.getcall(s0, time_t, vol)
        else:
            return V_mkt - self.getput(s0, time_t, vol)
    
    def grad_iv_prime(self, V_mkt, s0, time_t, vol):
        return -self.getvega(s0, time_t, vol)


    def get_iv(self, V_mkt, s0, time_t, is_call):
        tsize = [V_mkt.size(0), V_mkt.size(1)]
        tol = torch.full(tsize, 10**-3, dtype=torch.float64)
        vol_l = torch.full(tsize, 10**-8, dtype=torch.float64)
        vol_r = torch.full(tsize, 1., dtype=torch.float64)
        counter = 1000
        #print(vol_l, vol_r)
        if is_call:
            check = (self.getcall(s0, time_t, vol_l)*self.getcall(s0, time_t, vol_r)>0)
            #print(check)
        else:
            check = (self.getput(s0, time_t, vol_l)*self.getput(s0, time_t, vol_r)>0)
            #print(check)
        if check.all():
           print('no zero at the initial interval')
           return 0.
        else:
            vol = (vol_l + vol_r)/2
            epsilon = self.grad_iv(V_mkt, s0, time_t, vol, is_call)
            grad = self.grad_iv_prime(V_mkt, s0, time_t, vol)
            while (torch.abs(epsilon) > tol).all():

                condition00 = torch.abs(grad) > tol

                vol = torch.where(condition00, vol-epsilon/grad, vol)
                condition01 = torch.logical_and(torch.logical_and(vol-vol_r>0, vol_l-vol>0), condition00)
                vol = torch.where(condition01, (vol_l+vol_r)/2, vol)
                condition02 = torch.logical_and(self.grad_iv(V_mkt, s0, time_t, vol, is_call)*epsilon > 0, condition00)
                vol_l = torch.where(condition02, vol, vol_l)
                vol_r = torch.where(condition02, vol_r, vol)
                vol = torch.where(condition01, (vol_l+vol_r)/2, vol)

                condition10 = torch.logical_not(condition00)

                condition12 = torch.logical_and(self.grad_iv(V_mkt, s0, time_t, vol, is_call)*epsilon > 0, condition10)
                vol_l = torch.where(condition12, vol, vol_l)
                vol_r = torch.where(condition12, vol_r, vol)
                vol = torch.where(condition12, (vol_l+vol_r)/2, vol)

                epsilon = self.grad_iv(V_mkt, s0, time_t, vol, is_call)
                grad = self.grad_iv_prime(V_mkt, s0, time_t, vol)

                counter -= 1
                if counter == 0:
                    print('exit by counter')
                    return vol
            return vol


    """def get_iv(self, V_mkt, s0, time_t, is_call):
        tsize = [V_mkt.size(0), V_mkt.size(1)]
        tol = torch.full(tsize, 10**-4, dtype=torch.float64)
        vol_l = torch.full(tsize, 10**-8, dtype=torch.float64)
        vol_r = torch.full(tsize, 2, dtype=torch.float64)
        counter = 1000
        #print(vol_l, vol_r)
        if is_call:
            check = (self.getcall(s0, time_t, vol_l)*self.getcall(s0, time_t, vol_r)>0)
            #print(check)
        else:
            check = (self.getput(s0, time_t, vol_l)*self.getput(s0, time_t, vol_r)>0)
            #print(check)
        if check.all():
           print('no zero at the initial interval')
           return 0.
        else:
            vol = (vol_l + vol_r)/2
            epsilon = self.grad_iv(V_mkt, s0, time_t, vol, is_call)
            grad = self.grad_iv_prime(V_mkt, s0, time_t, vol)
            while (torch.abs(epsilon) > tol).all():
                if (torch.abs(grad) > tol).all():
                    vol -= epsilon / grad
                    if ((vol - vol_r)>0).all() or ((vol_l - vol)>0).all():
                        vol = (vol_l + vol_r)/2
                        if self.grad_iv(V_mkt, s0, time_t, vol, is_call)*epsilon > 0:
                            vol_l = vol
                        else:
                            vol_r = vol
                        vol = (vol_l + vol_r)/2
                else:
                    if self.grad_iv(V_mkt, s0, time_t, vol, is_call)*epsilon > 0:
                        vol_l = vol
                    else:
                        vol_r = vol
                    vol = (vol_l + vol_r)/2
                epsilon = self.grad_iv(V_mkt, s0, time_t, vol, is_call)
                grad = self.grad_iv_prime(V_mkt, s0, time_t, vol)
                counter -= 1
                if counter == 0:
                    print('exit by counter')
                    return vol
            return vol"""








    """def g_impvol(self, ValueMarket, sigma, T, S0, K, r, is_call=True):
        if is_call == True:
            return ValueMarket - self.getcall(
        return V_mkt - black_scholes_pv(sigma, S0, K, T, r, is_call)


    def g_impvol_prime(sigma, T, S0, K, r):
        return -black_scholes_vega(sigma, S0, K, T, r)

    def implied_vol(V_mkt, S0, K, T, r, is_call=True, tol=10**-4, sigma_l=10**-8, sigma_r=2):
        if g_impvol(V_mkt, sigma_l, T, S0, K, r, is_call)*\
            g_impvol(V_mkt, sigma_r, T, S0, K, r, is_call) > 0:
            print('no zero at the initial interval')
            return 0.
        else:
            sigma = (sigma_l + sigma_r) / 2
            epsilon = g_impvol(V_mkt, sigma, T, S0, K, r, is_call)
            grad = g_impvol_prime(sigma, T, S0, K, r)
            while abs(epsilon) > tol:   
                if abs(grad) > 1e-6:
                    sigma -= epsilon / grad
                    if sigma > sigma_r or sigma < sigma_l:
                        sigma = (sigma_l + sigma_r) / 2
                        if g_impvol(V_mkt, sigma_l, T, S0, K, r, is_call)*epsilon > 0:
                            sigma_l = sigma
                        else:
                            sigma_r = sigma
                        sigma = (sigma_l + sigma_r) / 2
                else:
                    if g_impvol(V_mkt, sigma_l, T, S0, K, r, is_call)*epsilon > 0:
                        sigma_l = sigma
                    else:
                        sigma_r = sigma
                    sigma = (sigma_l + sigma_r) / 2
            
                epsilon = g_impvol(V_mkt, sigma, T, S0, K, r, is_call)
                grad = g_impvol_prime(sigma, T, S0, K, r) 
            return sigma"""
    
