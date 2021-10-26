import numpy as np

class GordonUpdate:
    """
    Abstract data type for Gordon updates.
    """
    
    def __init__(self, sigma, delta):
        """
        Parameters
        ===
        sigma: noise standard deviation
        delta: oversampling ratio n/d
        """
        self.sigma = sigma
        self.delta = delta
        
    def PR_pop(self, alpha, beta):
        """
        One step population updates for Phase retrieval
        
        Parameters
        ===
        alpha: parallel component
        beta: orthogonal component

        Outputs
        ===
        alpha_next: updated parallel component
        beta_next: updated orthogonal component
        """
        phi = np.arctan(beta/alpha) # angle between iterate and truth
        alpha_next = 1 - 1/np.pi * (2 * phi - np.sin(2 * phi))
        
        beta_next = 2/np.pi * ( np.sin(phi) )**2
        return (alpha_next, beta_next)
    
    def GD_PR_pop(self, alpha, beta, eta):
        """
        One step population updates for Phase retrieval (with general step size)
        
        Parameters
        ===
        alpha: parallel component
        beta: orthogonal component

        Outputs
        ===
        alpha_next: updated parallel component
        beta_next: updated orthogonal component
        """
        phi = np.arctan(beta/alpha) # angle between iterate and truth
        alpha_next = (1 - 2*eta) * alpha + 2*eta * (1 - 1/np.pi * (2 * phi - np.sin(2 * phi)))
        
#         beta_next = np.sqrt((1 - 2*eta) * beta**2 + 2*eta*4/np.pi**2 * ( np.sin(phi) )**4)
        beta_T1 = (1 - 2 * eta) * beta + 2 * eta * 2/np.pi *  np.sin(phi)**2
        beta_next = np.abs(beta_T1)
        return (alpha_next, beta_next)

    def AM_PR_gordon(self, alpha, beta):
        """
        One step gordon updates for AM for Phase retrieval.

        Parameters
        ===
        alpha: parallel component
        beta: orthogonal component

        Outputs
        ===
        alpha_next: updated parallel component
        beta_next: updated orthogonal component
        """
        phi = np.arctan(beta/alpha) # angle between iterate and truth
        alpha_next = 1 - 1/np.pi * (2 * phi - np.sin(2 * phi))

        beta_next_T1 = 4/np.pi**2 * ( np.sin(phi) )**4
        beta_next_T2 = 1 - alpha_next**2 - beta_next_T1 + self.sigma**2
        beta_next_T3 = 1/(self.delta - 1) * beta_next_T2
        beta_next = np.sqrt(beta_next_T1 + beta_next_T3)

        return (alpha_next, beta_next)

    def GD_PR_gordon(self, alpha, beta, eta):
        """
        One step gordon updates for GD for Phase retrieval.

        Parameters
        ===
        alpha: parallel component
        beta: orthogonal component
        eta: stepsize

        Outputs
        ===
        alpha_next: updated parallel component
        beta_next: updated orthogonal component
        """
        phi = np.arctan(beta/alpha) # angle between iterate and truth
        
        ### update alpha
        alpha_T1 = (1 - 2 * eta) * alpha
        alpha_T2 = 2 * eta * (1 - 1/np.pi * (2 * phi - np.sin(2 * phi)))
        alpha_next = alpha_T1 + alpha_T2

        ### update beta
#         beta_T1 = (1 - 2 * eta) * beta**2
#         beta_T2 = 2 * eta * 4/np.pi**2 * ( np.sin(phi) )**4
        beta_T1 = (1 - 2 * eta) * beta + 2 * eta * 2/np.pi *  np.sin(phi)**2
        beta_T1 = beta_T1**2
        beta_T3_1 = alpha**2 + beta**2
        beta_T3_2 = -2 * alpha * (1 - 1/np.pi * (2 * phi - np.sin(2 * phi)))
        beta_T3_3 = -2 * beta * 2/np.pi * ( np.sin(phi) )**2
        beta_T3_4 = 1 + self.sigma**2
#         beta_T3 = 4 * eta**2/(self.delta - 1) * (beta_T3_1 + beta_T3_2\
#                                                 + beta_T3_3 + beta_T3_4)
        beta_T3 = 4 * eta**2/(self.delta) * (beta_T3_1 + beta_T3_2\
                                                + beta_T3_3 + beta_T3_4)
#         beta_next = np.sqrt(beta_T1 + beta_T2 + beta_T3)
        beta_next = np.sqrt(beta_T1 + beta_T3)
        
        return (alpha_next, beta_next)

    def AM_MLR_gordon(self, alpha, beta):
        """
        One step gordon updates for AM for MLR.

        Parameters
        ===
        alpha: parallel component
        beta: orthogonal component

        Outputs
        ===
        alpha_next: updated parallel component
        beta_next: updated orthogonal component
        """
        ### Preliminaries
        rho = beta/alpha 
        A = self.A(rho)
        B = self.B(rho)
        
        ### First calculate alpha
        alpha_next = 1 - A + B

        ### Next calculate beta
        beta_T1 = rho**2 * B**2
        beta_T2_1 = 1 + self.sigma**2
        beta_T2_2 = -alpha_next**2
        beta_T2_3 = -beta_T1
        beta_T2 = 1/(self.delta - 1) * (beta_T2_1 + beta_T2_2 + beta_T2_3)
        beta_next = np.sqrt(beta_T1 + beta_T2)

        return (alpha_next, beta_next)

    def GD_MLR_gordon(self, alpha, beta, eta):
        """
        One step gordon updates for GD for MLR.

        Parameters
        ===
        alpha: parallel component
        beta: orthogonal component
        eta: stepsize

        Outputs
        ===
        alpha_next: updated parallel component
        beta_next: updated orthogonal component
        """
        ### Preliminaries
        rho = beta/alpha
        A = self.A(rho)
        B = self.B(rho)

        ### First calculate alpha
        alpha_T1 = (1 - 2*eta) * alpha
        alpha_T2 = 2 * eta * (1 - A + B)
        alpha_next = alpha_T1 + alpha_T2

        ### Next calculate beta
        beta_T1 = (1 - 2*eta) * beta**2
        beta_T2 = 2 * eta * rho**2 * B**2
        beta_T3_1 = alpha**2 + beta**2 
        beta_T3_2 = -2*alpha*( 1 - A + B )
        beta_T3_3 = -2*beta * rho * B
        beta_T3_4 = 1 + self.sigma**2
#         beta_T3 = 4 * eta**2/(self.delta - 1) * (beta_T3_1 + beta_T3_2\
#                                                 + beta_T3_3 + beta_T3_4)
        beta_T3 = 4 * eta**2/(self.delta) * (beta_T3_1 + beta_T3_2\
                                                + beta_T3_3 + beta_T3_4)
        beta_next = np.sqrt(beta_T1 + beta_T2 + beta_T3)
        return (alpha_next, beta_next)

    def A(self, rho):
        """
        Computes the A function for MLR.

        Parameters
        ===
        rho: the value beta/alpha

        Outputs
        ===
        A_rho: the evaluation A(rho)
        """
        sqrt_term = np.sqrt(rho**2 + self.sigma**2 * (1 + rho**2))
        A_rho = 2/np.pi * np.arctan(sqrt_term)
        return A_rho

    def B(self, rho):
        """
        Computes the B function for MLR.

        Parameters
        ===
        rho: the value beta/alpha

        Outputs
        ===
        B_rho: the evaluation B(rho)
        """
        num_term = np.sqrt(rho**2 + self.sigma**2 * (1 + rho**2))
        denom_term = 1 + rho**2
        B_rho = 2/np.pi * num_term/denom_term
        return B_rho
