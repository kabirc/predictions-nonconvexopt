import numpy as np

class EmpiricalUpdate:
    """
    Abstract data type. To be extended by specific updates.
    """
    def __init__(self, thetastar, delta, sigma):
        """
        Parameters
        ===
        thetastar: ground truth
        delta: oversampling value (n/d)
        sigma: noise standard deviation
        """
        self.thetastar = thetastar
        self.delta = delta
        self.sigma = sigma
        
        ### Derived
        self.d = thetastar.shape[0]
        self.n = int(delta * self.d)

    def initialize(self, eps=0):
        """
        Initialize the iterative algorithm.  If eps is 0, then randomly
        initialize on the sphere.  Otherwise, initialize with the correlation
        given by eps.

        Parameters
        ===
        eps: initial correlation.  If 0, iniitalize randomly.  Otherwise,
        initialize with correlation eps

        Outputs
        ===
        None, sets the value self.thetainit
        """
        if eps == 0:
            # Random initialization
            thetainit = np.random.normal(0, 1, self.d)
            thetainit = thetainit/np.linalg.norm(thetainit)
            self.thetainit = thetainit
        else:
            # Initialize with correlation eps
            noise = np.random.normal(0, 1, self.d)
            perp_term = noise - np.dot(noise, self.thetastar) * self.thetastar
            perp_term = perp_term/np.linalg.norm(perp_term)
            thetainit = eps * self.thetastar + np.sqrt(1 - eps**2) * perp_term
            self.thetainit = thetainit

    def update_data(self):
        """
        Updates the data (since re-sampling is done).  All subclasses need the
        Gaussian matrix X. The response update is done in subclasses.

        Parameters 
        ===
        None

        Outputs
        ===
        None, sets the value self.X (of shape n, d)
        """
        self.X = np.random.normal(0, 1, (self.n, self.d))

    def get_alpha(self, theta):
        """
        Computes the parallel component at a given theta value. 

        Parameters
        ===
        theta: current iterate

        Outputs
        ===
        alpha: the value of the parallel component
        """
        alpha = np.dot(theta, self.thetastar)
        return alpha

    def get_beta(self, theta):
        """
        Computes the orthogonal component at a given theta value.

        Parameters
        ===
        theta: current iterate

        Outputs
        ===
        beta: value of the orthogonal component
        """
        orth_proj = theta - np.dot(theta, self.thetastar) * self.thetastar
        beta = np.linalg.norm(orth_proj)
        return beta

class SecondOrderEmpirical(EmpiricalUpdate):
    """
    Implements the updates for the second order methods.
    """
    def __init__(self, thetastar, delta, sigma):
        super().__init__(thetastar, delta, sigma)

    def update_data(self):
        """
        Updates the data for second order methods.  This includes precomputing
        the inverse Gram matrix.  Note: it is hard-coded in that the models are
        PR and MLR.

        Parameters
        ===
        None

        Outputs 
        ===
        None.  Updates the variables self.gram, self.gram_inv, self.y_PR, and
        self.y_MLR
        """
        super().update_data() # This updates self.X

        ### Update the inverses
        self.gram = np.dot(self.X.T, self.X) # gram matrix
        self.gram_inv = np.linalg.inv(self.gram) # inverse gram matrix

        ### Update the responses for the models
        noise = self.sigma * np.random.normal(0, 1, self.n)

        # update phase retrieval
        y_PR = np.abs(np.dot(self.X, self.thetastar)) + noise 
        self.y_PR = y_PR

        # update mixtures of linear regressions
        z = (-1)**np.random.binomial(1, 0.5, self.n) # labels
        y_MLR = z * np.dot(self.X, self.thetastar) + noise
        self.y_MLR = y_MLR

    def iterate_PR(self, theta):
        """
        Returns the next iterate according to the AM for phase retrieval
        update in Eq. ??

        Parameters
        ===
        theta: current iterate of shape (d)

        Outputs
        ===
        theta_next: next iterate of shape (d)
        """
        ### Term inside sum
        T1_1 = np.sign(np.dot(self.X, theta))
        T1 = T1_1 * self.y_PR
        sum_term = np.einsum('ij, i -> j', self.X, T1)

        # Multiply by inverse gram matrix
        theta_next = np.dot(self.gram_inv, sum_term)
        return theta_next

    def iterate_MLR(self, theta):
        """
        Returns the next iterate according to the AM for MLR 
        update in Eq. ??

        Parameters
        ===
        theta: current iterate of shape (d)

        Outputs
        ===
        theta_next: next iterate of shape (d)
        """
        ### Term inside sum
        T1_1 = np.sign(self.y_MLR * np.dot(self.X, theta))
        T1 = T1_1 * self.y_MLR
        sum_term = np.einsum('ij, i -> j', self.X, T1)

        # Multiply by inverse gram matrix
        theta_next = np.dot(self.gram_inv, sum_term)
        return theta_next

class FirstOrderEmpirical(EmpiricalUpdate):
    """
    Implements the updates for the first order methods.
    """
    def __init__(self, thetastar, delta, sigma):
        super().__init__(thetastar, delta, sigma)

    def update_data(self):
        """
        Updates the data for first order methods.  Note: it is 
        hard-coded in that the models are PR and MLR.

        Parameters
        ===
        None

        Outputs 
        ===
        None.  Updates the variables self.y_PR, and
        self.y_MLR
        """
        super().update_data() # This updates self.X

        ### Update the responses for the models
        noise = self.sigma * np.random.normal(0, 1, self.n)

        # update phase retrieval
        y_PR = np.abs(np.dot(self.X, self.thetastar)) + noise 
        self.y_PR = y_PR

        # update mixtures of linear regressions
        z = (-1)**np.random.binomial(1, 0.5, self.n) # labels
        y_MLR = z * np.dot(self.X, self.thetastar) + noise
        self.y_MLR = y_MLR

    def iterate_PR(self, theta, eta):
        """
        Returns the next iterate of subgradient descent for phase retrieval
        according to the update in Eq. ??

        Parameters
        ===
        theta: current iterate of shape (d)
        eta: step size

        Outputs
        ===
        theta_next: next iterate of shape (d)
        """
        ### Term inside sum
        T1_1 = np.abs(np.dot(self.X, theta)) - self.y_PR
        T1_2 = np.sign(np.dot(self.X, theta))
        T1 = T1_1 * T1_2
        sum_term = 2 * eta/self.n * np.einsum('ij, i -> j', self.X, T1)

        theta_next = theta - sum_term
        return theta_next

    def iterate_MLR(self, theta, eta):
        """
        Returns the next iterate of subgradient descent for MLR according to the
        update in Eq. ??

        Parameters
        ===
        theta: current iterate of shape (d)
        eta: stepsize

        Outputs
        === 
        theta_next: next iterate of shape (d)
        """
        Xtheta = np.dot(self.X, theta)
        T1_1 = np.sign(self.y_MLR * Xtheta)
        T1 = (T1_1 * Xtheta - self.y_MLR) * T1_1

        sum_term = 2 * eta/self.n * np.einsum('ij, i -> j', self.X, T1)
        theta_next = theta - sum_term
        return theta_next 
